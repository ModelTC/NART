# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/quantize/kl_divergence.py

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Find optimal scale for quantization by minimizing KL-divergence"""

try:
    from scipy import stats
except ImportError:
    stats = None

import numpy as np
import heapq


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError(
            "The discrete probability distribution is malformed. All entries are 0."
        )
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, "n_zeros=%d, n_nonzeros=%d, eps1=%f" % (
        n_zeros,
        n_nonzeros,
        eps1,
    )
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


eps = 1e-5


def new_hist(num_bins, min_val, max_val):
    return np.histogram(np.array([]), bins=num_bins, range=(min_val, max_val))


def update_hist(hist, edges, arr):
    min_val, max_val = edges[0], edges[-1]
    h, _ = np.histogram(
        np.clip(arr, min_val, max_val), bins=len(hist), range=(min_val, max_val)
    )
    hist += h


def adjust_zero_point(min_val, max_val, num_bins):
    max_val = max(0.0, float(max_val))
    min_val = min(0.0, float(min_val))
    if abs(min_val) < eps:
        return 0.0, max_val, 0
    elif abs(max_val) < eps:
        return min_val, 0.0, num_bins
    else:
        s = (max_val - min_val) / num_bins
        zq = round(-min_val / s)
        min_p, max_p = -zq * s, (num_bins - zq) * s
        s_new = max(-min_val / zq, max_val / (num_bins - zq))
        min_p, max_p = -zq * s_new, (num_bins - zq) * s_new
        assert min_p < min_val + eps
        assert max_p > max_val - eps
        assert min(abs(min_p - min_val), abs(max_p - max_val)) < eps
        return min_p, max_p, zq


def kl_divergence_scale(arr, num_bins, num_quantized_bins=255):
    """Given a tensor, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.
    Ref:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)

    min_val = np.min(arr)
    max_val = np.max(arr)

    min_val, max_val, zq = adjust_zero_point(min_val, max_val, num_bins)
    hist, hist_edges = new_hist(num_bins, min_val, max_val)
    update_hist(hist, hist_edges, arr)

    return min_kl_range(hist, hist_edges, zq, num_quantized_bins)


def min_kl_range(hist, hist_edges, zero_bin_idx, num_quantized_bins=255):
    assert len(hist) + 1 == len(hist_edges)
    num_bins = len(hist)
    res_queue = []

    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.

    for p_bin_idx_start in range(zero_bin_idx + 1):
        st = max(zero_bin_idx, p_bin_idx_start + num_quantized_bins)
        if st % 2 == p_bin_idx_start % 2:
            st += 1
        for p_bin_idx_stop in range(st, len(hist_edges), 2):
            sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            assert p.size % 2 == 1
            assert p.size >= num_quantized_bins
            # put left outlier count in p[0]
            left_outlier_count = np.sum(hist[0:p_bin_idx_start])
            p[0] += left_outlier_count
            # put right outlier count in p[-1]
            right_outlier_count = np.sum(hist[p_bin_idx_stop:])
            p[-1] += right_outlier_count
            # is_nonzeros[k] indicates whether hist[k] is nonzero
            is_nonzeros = (p != 0).astype(np.int32)

            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = sliced_nd_hist.size // num_quantized_bins
            # merge hist into num_quantized_bins bins
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[
                num_quantized_bins * num_merged_bins :
            ].sum()
            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                if j == num_quantized_bins - 1:
                    stop = len(is_nonzeros)
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = float(quantized_bins[j]) / float(norm)
            q[p == 0] = 0
            p = _smooth_distribution(p)
            # There is a chance that q is an invalid probability distribution.
            try:
                q = _smooth_distribution(q)
            except ValueError:
                continue
            d = stats.entropy(p, q)
            if np.isnan(d):
                continue

            # negative
            heapq.heappush(
                res_queue,
                (-d, (hist_edges[p_bin_idx_start], hist_edges[p_bin_idx_stop])),
            )
            if len(res_queue) > 30:
                heapq.heappop(res_queue)

    res_queue = sorted([(-nd, rg) for nd, rg in res_queue])
    return res_queue[0]
