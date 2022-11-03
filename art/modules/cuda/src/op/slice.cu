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

#include <stdlib.h>
#include <string.h>

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t axis;
    size_t slice_len;
    uint32_t *slice_points;
} op_slice_t;

op_slice_t *op_cuda_slice_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_slice_t *res = (op_slice_t *)malloc(sizeof(op_slice_t));
    memset(res, 0, sizeof(op_slice_t));
    return res;
}

void op_cuda_slice_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_SLICE_AXIS, dtUINT32, &((op_slice_t *)op)->axis));
    CHECK(op_setting_array_get(
        op, SETTING_SLICE_POINT, dtUINT32, &((op_slice_t *)op)->slice_len,
        &((op_slice_t *)op)->slice_points));
}

void op_cuda_slice_tp_destroy(op_t *op) { (void)op; }

void op_cuda_slice_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_cuda_slice_run(op_t *op)
{
    int i;
    size_t width;
    uint32_t axis = ((op_slice_t *)op)->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_size = shape_part_count(&op->input_tensors[0]->shape, axis) / ch;
    size_t outer_size = shape_count(&op->input_tensors[0]->shape) / ch / inner_size;
    const uint8_t *src = (uint8_t const *)(mem_data(op->input_tensors[0]->mem));
    size_t src_stride = inner_size * op->input_tensors[0]->shape.dim[axis] * sizeof(float);
    for (i = 0; i < op->output_size; ++i) {
        width = inner_size * op->output_tensors[i].shape.dim[axis] * sizeof(float);

        CUDA_CHECK(cudaMemcpy2DAsync(
            mem_data(op->output_tensors[i].mem), width, src, src_stride, width, outer_size,
            cudaMemcpyDeviceToDevice, CUDA_WORKSPACE_STREAM(op->workspace)));

        // increment source
        src += width;
    }
}

void op_cuda_slice_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_slice_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
