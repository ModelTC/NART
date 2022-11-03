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
    uint32_t axis;
} op_concat_t;

op_concat_t *op_cuda_concat_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_concat_t *res = (op_concat_t *)malloc(sizeof(op_concat_t));
    memset(res, 0, sizeof(op_concat_t));
    return res;
}

void op_cuda_concat_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_CONCAT_AXIS, dtUINT32, &((op_concat_t *)op)->axis));
}

void op_cuda_concat_tp_destroy(op_t *op) { (void)op; }

void op_cuda_concat_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_cuda_concat_run(op_t *op)
{
    int i;
    op_concat_t *concat_op = (op_concat_t *)op;
    size_t axis = concat_op->axis;
    size_t inner_num = 0;
    size_t outer_num = 0;
    for (int i = 0; i < op->input_size; ++i) {
        int ch = op->input_tensors[i]->shape.dim[axis];
        if (ch) {
            inner_num = shape_part_count(&op->input_tensors[i]->shape, concat_op->axis) / ch;
            outer_num = shape_count(&op->input_tensors[i]->shape) / inner_num / ch;
            break;
        }
    }
    CHECK(inner_num != 0);
    CHECK(outer_num != 0);

    const float *input = NULL;
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    for (i = 0; i < op->input_size; ++i) {
        int ch = op->input_tensors[i]->shape.dim[axis];
        if (0 == ch)
            continue;
        input = (const float *)mem_data(op->input_tensors[i]->mem);
        size_t width = inner_num * ch * datatype_sizeof(op->output_tensors[0].dtype);
        size_t dpitch = inner_num * op->output_tensors[0].shape.dim[axis]
            * datatype_sizeof(op->output_tensors[0].dtype);
        cudaMemcpy2DAsync(
            output, dpitch, input, width, width, outer_num, cudaMemcpyDeviceToDevice,
            CUDA_WORKSPACE_STREAM(op->workspace));
        output += inner_num * op->input_tensors[i]->shape.dim[axis];
    }
}

void op_cuda_concat_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_concat_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
