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

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../utils/sgemm.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
} op_matmul_t;
op_matmul_t *op_default_matmul_tp_alloc(workspace_t *ws);
void op_default_matmul_tp_config(op_t *op);
void op_default_matmul_tp_destroy(op_t *op);
void op_default_matmul_tp_dealloc(op_t *op);
void op_default_matmul_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif

op_matmul_t *op_default_matmul_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_matmul_t *res = (op_matmul_t *)malloc(sizeof(op_matmul_t));
    memset(res, 0, sizeof(op_matmul_t));
    return res;
}

void op_default_matmul_tp_config(op_t *op) { (void)op; }

void op_default_matmul_tp_destroy(op_t *op) { (void)op; }

void op_default_matmul_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void cpu_matmul(
    float *output_0, const float *input_0, const float *input_1, size_t m, size_t n, size_t k)
{
    size_t i, j, l;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            for (l = 0; l < k; ++l) {
                output_0[i * n + j] += input_0[i * k + l] * input_1[l * n + j];
            }
        }
    }
}

static size_t calc_offset(shape_t *in_shape, shape_t *out_shape, size_t index)
{
    /*
    for broadcasted matmul, calculate offset for input, given the offset for output.
    for example, given two inputs:
    1 x  8 x 32 x 256
    32 x 1 x 256 x 8
    the output shape is:
    32 x 8 x 32 x 8
    there are 256 batch in total, when given offset 60 (the 61-th batch), how we calculate the
    offset for both inputs? first calculate it's index, 60 = 7 * 8 + 4, then it's index is [7, 4].
    map [7, 4] to input-0, we get [1, 4], means the offset is 4.
    map [7, 4] to input-1, we get [7, 1], mean the offset is 7.
    */
    int32_t axis = 0;
    size_t offset = 0;
    // the
    size_t count_after = 1;
    for (axis = out_shape->dim_size - 3; axis >= 0; axis -= 1) {
        // idx_axis is the index at the axis-th dimension.
        size_t idx_axis = index % out_shape->dim[axis];
        index = index / out_shape->dim[axis];
        // to get the the index in input shape, we divide it by input's length on axis-th dimension.
        offset += (idx_axis % in_shape->dim[axis]) * count_after;
        count_after *= in_shape->dim[axis];
    }
    return offset;
}

static void op_default_matmul_run(op_t *op)
{
    size_t i;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(float) * shape_count(&op->output_tensors[0].shape));
    shape_t shape_i1 = op->input_tensors[0]->shape;
    shape_t shape_i2 = op->input_tensors[1]->shape;
    if (shape_i1.dim_size == shape_i2.dim_size) {
        // eg: 2D * 2D , 3D * 3D ...
        size_t size = shape_i1.dim_size;
        size_t m = op->input_tensors[0]->shape.dim[size - 2];
        size_t k = op->input_tensors[0]->shape.dim[size - 1];
        size_t n = op->input_tensors[1]->shape.dim[size - 1];
        size_t count_o = shape_count(&op->output_tensors[0].shape) / m / n;
        for (i = 0; i < count_o; i += 1) {
            size_t offset_0
                = calc_offset(&op->input_tensors[0]->shape, &op->output_tensors[0].shape, i);
            size_t offset_1
                = calc_offset(&op->input_tensors[1]->shape, &op->output_tensors[0].shape, i);
            // printf("offset0=%u, offset1=%u\n", offset_0, offset_1);
            // sgemm_AxB(m, n, k, input_0, input_1, output_0);
            cpu_matmul(output_0, input_0 + offset_0 * m * k, input_1 + offset_1 * k * n, m, n, k);
            output_0 += m * n;
        }

    } else {
        // eg: 3D * 2D , 4D * 2D ...
        size_t size1 = shape_i1.dim_size;
        size_t size2 = shape_i2.dim_size;
        size_t m = op->input_tensors[0]->shape.dim[size1 - 2];
        size_t k = op->input_tensors[0]->shape.dim[size1 - 1];
        size_t n = op->input_tensors[1]->shape.dim[size2 - 1];
        size_t count_o = shape_count(&op->output_tensors[0].shape) / m / n;
        for (i = 0; i < count_o; i += m * n) {
            // sgemm_AxB(m, n, k, input_0, input_1, output_0);
            cpu_matmul(output_0, input_0, input_1, m, n, k);
            output_0 += m * n;
            input_0 += m * k;
        }
    }
}

void op_default_matmul_tp_prepare(op_t *op)
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
        op->run_func = op_default_matmul_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
