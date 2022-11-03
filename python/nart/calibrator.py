#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse
import numpy as np
from google.protobuf.text_format import Merge
import json
import logging
import re
import copy
import math
from nart import asym_kl

NUM_ASYM_ACTIVATE_BINS = 483
NUM_ASYM_WEIGHT_BINS = 321


class ClbConfig:
    config = None

    def __init__(self):
        self.input_dir = None
        self.bit_width = 8
        self.use_gpu = False
        self.device = 0
        self.data_num = -1
        self.symmetric = False
        self.kl_divergence = False
        self.bins = 2048

    @staticmethod
    def get_config():
        if ClbConfig.config == None:
            ClbConfig.config = ClbConfig()
        return ClbConfig.config


def str2var(name):
    patt = re.compile("[^a-zA-Z0-9]")
    name = re.sub(patt, "_", name)
    if name[0].isdigit():
        name = "_" + name
    return name


def quant_weights(args, model):
    ret = {}
    for i in model.layer:
        if (
            i.type == "Convolution"
            or i.type == "InnerProduct"
            or i.type == "PReLU"
            or i.type == "Deconvolution"
        ):
            wmax = max(np.array(i.blobs[0].data).max(), 0.0)
            wmin = min(np.array(i.blobs[0].data).min(), 0.0)

            if not args.symmetric and args.kl_divergence:
                _, (wmin, wmax) = asym_kl.kl_divergence_scale(
                    np.array(i.blobs[0].data),
                    NUM_ASYM_WEIGHT_BINS,
                    2**args.bit_width - 1,
                )
                print(i.name, _, wmin, wmax)

            ret[f"{i.name}_weight_0"] = {
                "max": wmax,
                "min": wmin,
                "bit": args.bit_width,
                "type": "biased",
            }

            # if i.convolution_param.bias_term or i.inner_product_param.bias_term:
            #    bmax = max(np.array(i.blobs[1].data).max(), 0)
            #    bmin = min(np.array(i.blobs[1].data).min(), 0)
            #    ret[f'{i.name}_bias'] = {'max':bmax, 'min':bmin}
        elif i.type == "BatchNorm" or i.type == "Scale":
            print(f"emmmm {i.type} not supported yet!")

    if args.symmetric:
        for k in ret:
            imax = float(ret[k]["max"])
            imin = float(ret[k]["min"])
            ret[k]["max"] = max(abs(imax), abs(imin))
            ret[k]["min"] = -ret[k]["max"]
            ret[k]["type"] = "symmetric"

    return ret


def get_inputs(model):
    inputs = []
    inputs.extend(model.input)
    return inputs


def input_data_generator(input_dir, inputs, data_num=-1):
    cnt = 0
    while True:
        data = {}
        try:
            for i in inputs:
                f = open(f"{input_dir}/{str2var(i)}/{cnt}.bin", "rb")
                data[i] = np.fromfile(f"{input_dir}/{str2var(i)}/{cnt}.bin", "float32")
            print(f"processing {cnt}.bin")
            yield data
            cnt += 1
            if data_num != -1 and cnt >= data_num:
                return
        except FileNotFoundError:
            print(f"total: {cnt}")
            return


def kl_divergence(dist_a, dist_b):
    nonzero_inds = dist_a != 0
    return np.sum(
        dist_a[nonzero_inds] * np.log(dist_a[nonzero_inds] / dist_b[nonzero_inds])
    )


def threshold_distribution(dist, target_bin=128):
    target_threshold = target_bin
    min_kl_divergence = 1000
    length = len(dist)

    quantize_dist = np.zeros(target_bin)

    threshold_sum = 0.0
    threshold_sum = sum(dist[target_bin:])

    for threshold in range(target_bin, length):
        ref_dist = copy.deepcopy(dist[:threshold])
        ref_dist[threshold - 1] = ref_dist[threshold - 1] + threshold_sum
        threshold_sum = threshold_sum - dist[threshold]

        quantize_dist = np.zeros(target_bin)
        num_per_bin = threshold / target_bin
        for i in range(0, target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            left_upper = (int)(math.ceil(start))
            if left_upper > start:
                left_scale = left_upper - start
                quantize_dist[i] += left_scale * dist[left_upper - 1]

            right_lower = (int)(math.floor(end))
            if right_lower < end:
                right_scale = end - right_lower
                quantize_dist[i] += right_scale * dist[right_lower]

            for j in range(left_upper, right_lower):
                quantize_dist[i] += dist[j]

        dequant_dist = np.zeros(threshold, dtype=np.float32)

        for i in range(0, target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            count = 0

            left_upper = (int)(math.ceil(start))
            left_scale = 0.0
            if left_upper > start:
                left_scale = left_upper - start
                if dist[left_upper - 1] != 0:
                    count += left_scale

            right_lower = (int)(math.floor(end))
            right_scale = 0.0
            if right_lower < end:
                right_scale = end - right_lower
                if dist[right_lower] != 0:
                    count += right_scale

            for j in range(left_upper, right_lower):
                if dist[j] != 0:
                    count = count + 1

            expand_value = quantize_dist[i] / count

            if left_upper > start:
                if dist[left_upper - 1] != 0:
                    dequant_dist[left_upper - 1] += expand_value * left_scale
            if right_lower < end:
                if dist[right_lower] != 0:
                    dequant_dist[right_lower] += expand_value * right_scale
            for j in range(left_upper, right_lower):
                if dist[j] != 0:
                    dequant_dist[j] += expand_value

        kl = kl_divergence(ref_dist, dequant_dist)

        if kl < min_kl_divergence:
            min_kl_divergence = kl
            target_threshold = threshold
    return target_threshold


def quant_activation(args, model, net):
    # step 1: calc max, min for each input_data
    act_dict, act_ext = {}, {}
    inputs = get_inputs(model)
    for k in net.blobs:
        act_dict[k] = {"max": [], "min": []}
        act_ext[k] = {"mean": [], "std": []}

    for data in input_data_generator(args.input_dir, inputs, data_num=args.data_num):
        for k in data:
            net.blobs[k].data[:] = data[k].reshape(net.blobs[k].data.shape)
        net.forward()

        for k in net.blobs:
            data = np.array(net.blobs[k].data)
            act_dict[k]["max"].append(net.blobs[k].data.max())
            act_dict[k]["min"].append(net.blobs[k].data.min())
            act_ext[k]["mean"].append(data.mean())
            act_ext[k]["std"].append(data.std())
            act_dict[k]["bit"] = args.bit_width
            act_dict[k]["type"] = "biased"

    # step 2: gather statistics of activations' dist
    if args.symmetric:
        if args.kl_divergence:
            blobs_dist_dict = {}
            for k in net.blobs:
                b_max = np.max(act_dict[k]["max"])
                b_min = np.min(act_dict[k]["min"])
                blobs_dist_dict[k] = {
                    "dist": np.zeros(args.bins).astype("float64"),
                    "max": max(b_max, abs(b_min)),
                    "min": -max(b_max, abs(b_min)),
                }

            for data in input_data_generator(
                args.input_dir, inputs, data_num=args.data_num
            ):
                for k in data:
                    net.blobs[k].data[:] = data[k].reshape(net.blobs[k].data.shape)
                net.forward()
                for k in net.blobs:
                    bins, c = np.histogram(
                        np.abs(net.blobs[k].data[:]),
                        bins=args.bins,
                        range=(1e-10, blobs_dist_dict[k]["max"]),
                    )
                    blobs_dist_dict[k]["dist"] += bins

            for k in net.blobs:
                blobs_dist_dict[k]["dist"] = (
                    blobs_dist_dict[k]["dist"] / blobs_dist_dict[k]["dist"].sum()
                )
                thr = threshold_distribution(blobs_dist_dict[k]["dist"])
                act_dict[k]["max"] = (thr + 0.5) * (
                    blobs_dist_dict[k]["max"] / args.bins
                )
                act_dict[k]["min"] = -act_dict[k]["max"]

        for k in act_dict:
            amax = max(np.average(act_dict[k]["max"]), 0)
            amin = min(np.average(act_dict[k]["min"]), 0)
            act_dict[k]["max"] = max(abs(amax), abs(amin))
            act_dict[k]["min"] = -act_dict[k]["max"]
            act_dict[k]["type"] = "symmetric"
    elif not args.symmetric and args.kl_divergence:
        for k in act_dict:
            min_val, max_val = min(act_dict[k]["min"]), max(act_dict[k]["max"])
            min_val, max_val, act_ext[k]["zq"] = asym_kl.adjust_zero_point(
                min_val, max_val, NUM_ASYM_ACTIVATE_BINS
            )
            act_ext[k]["hist"], act_ext[k]["hist_edges"] = asym_kl.new_hist(
                NUM_ASYM_ACTIVATE_BINS, min_val, max_val
            )
        for data in input_data_generator(
            args.input_dir, inputs, data_num=args.data_num
        ):
            for k in data:
                net.blobs[k].data[:] = data[k].reshape(net.blobs[k].data.shape)
            net.forward()
            for k in net.blobs:
                asym_kl.update_hist(
                    act_ext[k]["hist"],
                    act_ext[k]["hist_edges"],
                    np.array(net.blobs[k].data),
                )
        for k in act_dict:
            _, (act_dict[k]["min"], act_dict[k]["max"]) = asym_kl.min_kl_range(
                act_ext[k]["hist"],
                act_ext[k]["hist_edges"],
                act_ext[k]["zq"],
                2**args.bit_width - 1,
            )
            act_dict[k]["min"], act_dict[k]["max"] = map(
                float, (act_dict[k]["min"], act_dict[k]["max"])
            )
            print(k, _, act_dict[k]["min"], act_dict[k]["max"])
    else:
        for k in act_dict:

            # calc final max/min
            amin, amax = np.array(act_dict[k]["min"]), np.array(act_dict[k]["max"])
            act_dict[k]["min"] = float(amin.mean())
            act_dict[k]["max"] = float(amax.mean())

            # ema
            # fmax = act_dict[k]['max'][0]
            # fmin = act_dict[k]['min'][0]
            # alpha = 0.9
            # for i in range(len(act_dict[k]['max'])):
            #     fmax = alpha * fmax + (1 - alpha) * act_dict[k]['max'][i]
            #     fmin = alpha * fmin + (1 - alpha) * act_dict[k]['min'][i]
            # act_dict[k]['max'] = fmax
            # act_dict[k]['min'] = fmin

            # act_dict[k]['type'] = 'biased'

    # adjust softmax/sigmoid
    for l in model.layer:
        if l.type in ["Softmax", "Sigmoid"]:
            act_dict[l.top[0]]["max"] = 1.0
            act_dict[l.top[0]]["min"] = 0.0

    # step 3: generate max, min
    return act_dict


def quant_bias(args, model, param):
    bias_dict = {}
    for l in model.layer:
        if (
            (l.type == "Convolution" or l.type == "Deconvolution")
            and l.convolution_param.bias_term
        ) or (l.type == "InnerProduct" and l.inner_product_param.bias_term):
            wmax = param[f"{l.name}_weight_0"]["max"]
            wmin = param[f"{l.name}_weight_0"]["min"]
            imax = param[f"{l.bottom[0]}"]["max"]
            imin = param[f"{l.bottom[0]}"]["min"]

            if args.symmetric:
                walpha = (wmax - wmin) / (2**args.bit_width - 2)
                ialpha = (imax - imin) / (2**args.bit_width - 2)
            else:
                walpha = (wmax - wmin) / (2**args.bit_width - 1)
                ialpha = (imax - imin) / (2**args.bit_width - 1)
            param[f"{l.name}_weight_1"] = {
                "alpha": walpha * ialpha,
                "zero_point": 0,
                "type": "symmetric",
                "bit": 32,
            }

    return bias_dict


def calibrate(prototxt, caffemodel, args):
    import caffe

    # set gpu
    if args.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(args.device)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # gen caffe model
    logger.debug("start gen caffe model")
    proto = caffe.proto.caffe_pb2.NetParameter()
    with open(prototxt, "rt") as f:
        Merge(f.read(), proto)
    weight = caffe.proto.caffe_pb2.NetParameter()
    with open(caffemodel, "rb") as f:
        weight.ParseFromString(f.read())
    model = proto
    for i in model.layer:
        for j in weight.layer:
            if i.name == j.name:
                i.blobs.extend(j.blobs)
                break

    logger.debug("finish gen caffe model")

    logger.debug("start gen caffe net")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    logger.debug("finish gen caffe net")

    param = {}
    logger.debug("start quant weights")
    param.update(quant_weights(args, model))
    logger.debug("finish quant weights")

    logger.debug("start activation weights")
    param.update(quant_activation(args, model, net))
    logger.debug("finish activation weights")

    param.update(quant_bias(args, model, param))
    return param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--prototxt", help="prototxt", default="model/rel.prototxt"
    )
    parser.add_argument("-b", "--model", help="caffemodel", default="model/model.bin")
    parser.add_argument("-i", "--input_dir", help="input dir", default="model/input/")
    parser.add_argument("-n", "--data_num", type=int, help="input data num", default=-1)
    parser.add_argument(
        "-s", "--symmetric", help="use symmetric quantization", action="store_true"
    )
    parser.add_argument(
        "-k", "--kl_divergence", help="use kl-divergence", action="store_true"
    )
    parser.add_argument("--bins", type=int, help="bins", default=2048)
    parser.add_argument("-g", "--use_gpu", help="use gpu", action="store_true")
    parser.add_argument("-d", "--device", type=int, help="which gpu", default=0)
    parser.add_argument("-B", "--bit_width", type=int, help="bit-width", default=8)
    parser.add_argument(
        "-o", "--output", type=str, help="output file name", default="max_min.param"
    )
    args = parser.parse_args()

    # set gpu
    import caffe

    if args.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(args.device)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # gen caffe model
    logger.debug("start gen caffe model")
    proto = caffe.proto.caffe_pb2.NetParameter()
    with open(args.prototxt, "rt") as f:
        Merge(f.read(), proto)
    weight = caffe.proto.caffe_pb2.NetParameter()
    with open(args.model, "rb") as f:
        weight.ParseFromString(f.read())
    model = proto
    for i in model.layer:
        for j in weight.layer:
            if i.name == j.name:
                i.blobs.extend(j.blobs)
                break

    logger.debug("finish gen caffe model")

    logger.debug("start gen caffe net")
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    logger.debug("finish gen caffe net")

    param = {}
    logger.debug("start quant weights")
    param.update(quant_weights(args, model))
    logger.debug("finish quant weights")

    logger.debug("start activation")
    param.update(quant_activation(args, model, net))
    logger.debug("finish activation")

    param.update(quant_bias(args, model, param))

    with open(args.output, "wt") as f:
        json.dump(param, f, indent=4)


if __name__ == "__main__":
    main()
