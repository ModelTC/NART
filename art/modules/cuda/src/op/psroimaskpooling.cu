/*
 * Copyright 2022 SenseTime Group Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    float spatial_scale;
    uint32_t output_dim;
    uint32_t group_size;
    float roi_scale;
    float bin_scale;
} op_psroimaskpooling_t;

op_psroimaskpooling_t *op_cuda_psroimaskpooling_tp_alloc(workspace_t *ws);
void op_cuda_psroimaskpooling_tp_config(op_t *op);
void op_cuda_psroimaskpooling_tp_prepare(op_t *op);
void op_cuda_psroimaskpooling_tp_destroy(op_t *op);
void op_cuda_psroimaskpooling_tp_dealloc(op_t *op);

#ifdef __cplusplus
}
#endif

op_psroimaskpooling_t *op_cuda_psroimaskpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_psroimaskpooling_t *res = (op_psroimaskpooling_t *)malloc(sizeof(op_psroimaskpooling_t));
    memset(res, 0, sizeof(op_psroimaskpooling_t));
    return res;
}

void op_cuda_psroimaskpooling_tp_dealloc(op_t *op)
{
    if (op != NULL) {
        free(op);
    }
}

void op_cuda_psroimaskpooling_tp_config(op_t *op)
{
    op_psroimaskpooling_t *pool_op = (op_psroimaskpooling_t *)op;
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_SPATIAL_SCALE, dtFLOAT32, &pool_op->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_ROI_SCALE, dtFLOAT32, &pool_op->roi_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_BIN_SCALE, dtFLOAT32, &pool_op->bin_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_GROUP_SIZE, dtUINT32, &pool_op->group_size));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_OUTPUT_DIM, dtUINT32, &pool_op->output_dim));
}

template <typename Dtype>
__global__ void PSROIMaskPoolingForward(
    const int nthreads, const Dtype *bottom_data, const Dtype spatial_scale, const Dtype roi_scale,
    const Dtype bin_scale, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const Dtype *bottom_rois, const int output_dim,
    const int group_size, Dtype *top_data, const int shape)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        bottom_rois += n * shape;

        const int roi_batch_ind = static_cast<int>(bottom_rois[0]);

        const Dtype x1 = bottom_rois[1];
        const Dtype y1 = bottom_rois[2];
        const Dtype x2 = bottom_rois[3];
        const Dtype y2 = bottom_rois[4];

        // TODO(kun): The following equations for calculating w and h
        // are not technically correct.
        Dtype w = x2 - x1;
        Dtype h = y2 - y1;

        Dtype xc = (x1 + x2) * Dtype(0.5);
        Dtype yc = (y1 + y2) * Dtype(0.5);

        // Rescale RoIs with regard to roi_scale
        Dtype xx1 = xc - w * roi_scale * Dtype(0.5);
        Dtype xx2 = xc + w * roi_scale * Dtype(0.5);
        Dtype yy1 = yc - h * roi_scale * Dtype(0.5);
        Dtype yy2 = yc + h * roi_scale * Dtype(0.5);

        Dtype roi_start_w = round(xx1) * spatial_scale;
        Dtype roi_start_h = round(yy1) * spatial_scale;
        Dtype roi_end_w = (round(xx2) + Dtype(1.)) * spatial_scale;
        Dtype roi_end_h = (round(yy2) + Dtype(1.)) * spatial_scale;

        // Force too small ROIs to be 1 x 1
        Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(0.1)); // avoid 0
        Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(0.1));

        // Compute w and h at bottom
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        Dtype delta_h = (bin_size_h * bin_scale - bin_size_h) * Dtype(0.5);
        Dtype delta_w = (bin_size_w * bin_scale - bin_size_w) * Dtype(0.5);

        int hstart = static_cast<int>(
            floor((static_cast<Dtype>(ph) * bin_size_h + roi_start_h) - delta_h));
        int wstart = static_cast<int>(
            floor((static_cast<Dtype>(pw) * bin_size_w + roi_start_w) - delta_w));
        int hend = static_cast<int>(
            ceil((static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h) + delta_h));
        int wend = static_cast<int>(
            ceil((static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w) + delta_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        Dtype out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                out_sum += bottom_data[bottom_index];
            }
        }

        Dtype bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? Dtype(0.) : (out_sum / bin_area);
    }
}

static void op_cuda_psroimaskpooling_run_fp32(op_t *op)
{
    op_psroimaskpooling_t *pool_op = (op_psroimaskpooling_t *)op;
    tensor_t *output = &op->output_tensors[0];
    tensor_t *feature = op->input_tensors[0];
    tensor_t *rois = op->input_tensors[1];

    const int channel_in = feature->shape.dim[1];
    const int height_in = feature->shape.dim[2];
    const int width_in = feature->shape.dim[3];
    // const int num_rois = rois->shape.dim[0];
    const int roi_shape = rois->shape.dim[1];
    const int group_size = pool_op->group_size;
    float *feature_data = (float *)mem_data(feature->mem);
    float *rois_data = (float *)(mem_data(rois->mem));
    float *output_data = (float *)(mem_data(output->mem));

    size_t output_count = shape_count(&(output->shape));
    PSROIMaskPoolingForward<float>
        <<<(output_count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_count, feature_data, pool_op->spatial_scale, pool_op->roi_scale,
            pool_op->bin_scale, channel_in, height_in, width_in, group_size, group_size, rois_data,
            pool_op->output_dim, group_size, output_data, roi_shape);
}

void op_cuda_psroimaskpooling_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    CHECK(op->input_tensors[1]->dtype == dtFLOAT32);
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_psroimaskpooling_run_fp32;
        break;
    default:
        LOG_error("PSROIMaskPooling op on dtype `%d` not implemented", op->input_tensors[0]->dtype);
        break;
    }
}

void op_cuda_psroimaskpooling_tp_destroy(op_t *op)
{
    // nothing to do
    (void)op;
}
