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
#include "art/op_tp.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    op_t o;
} op_mul_t;
op_mul_t *op_default_mul_tp_alloc(workspace_t *ws);
void op_default_mul_tp_config(op_t *op);
void op_default_mul_tp_destroy(op_t *op);
void op_default_mul_tp_dealloc(op_t *op);
void op_default_mul_tp_prepare(op_t *op);
#ifdef __cplusplus
} // extern "C"
#endif

op_mul_t *op_default_mul_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_mul_t *res = (op_mul_t *)malloc(sizeof(op_mul_t));
    memset(res, 0, sizeof(op_mul_t));
    return res;
}

void op_default_mul_tp_config(op_t *op) { (void)op; }

void op_default_mul_tp_destroy(op_t *op) { (void)op; }

void op_default_mul_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T> static void op_default_mul_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    int i, j;
    const T *input_0 = (const T *)mem_cpu_data(op->input_tensors[0]->mem);
    const T *input_1 = (const T *)mem_cpu_data(op->input_tensors[1]->mem);
    T *output_0 = (T *)mem_cpu_data(op->output_tensors[0].mem);
    if (tensor_shape_eq(&op->input_tensors[0]->shape, &op->input_tensors[1]->shape)) {
        for (i = 0; i < (int)count; ++i) {
            output_0[i] = input_0[i] * input_1[i];
        }
    } else {
        shape_t shape = op->output_tensors[0].shape;
        shape_t shape1 = op->input_tensors[0]->shape;
        shape_t shape2 = op->input_tensors[1]->shape;
        uint32_t *shape_dim = shape.dim;
        uint32_t *shape_dim1 = shape1.dim;
        uint32_t *shape_dim2 = shape2.dim;
        for (i = 0; i < shape1.dim_size; ++i) {
            shape1.dim[shape.dim_size - 1 - i] = shape1.dim[shape1.dim_size - 1 - i];
        }
        for (; i < shape.dim_size; ++i) {
            shape1.dim[shape.dim_size - 1 - i] = 1;
        }
        for (i = 0; i < shape2.dim_size; ++i) {
            shape2.dim[shape.dim_size - 1 - i] = shape2.dim[shape2.dim_size - 1 - i];
        }
        for (; i < shape.dim_size; ++i) {
            shape2.dim[shape.dim_size - 1 - i] = 1;
        }
        for (i = 0; i < (int)count; ++i) {
            uint32_t p = 1, p1 = 1, p2 = 1;
            uint32_t id1 = 0, id2 = 0;
            for (j = shape.dim_size - 1; j >= 0; j--) {
                uint32_t tmp = (i / p) % (shape_dim[j]);
                id1 += (shape_dim1[j] == 1 ? 0 : tmp) * p1;
                id2 += (shape_dim2[j] == 1 ? 0 : tmp) * p2;
                p1 *= shape_dim1[j];
                p2 *= shape_dim2[j];
                p *= shape_dim[j];
            }
            output_0[i] = input_0[id1] * input_1[id2];
        }
    }
}

void op_default_mul_tp_prepare(op_t *op)
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
        op->run_func = op_default_mul_run<float>;
        break;
    case dtINT64:
        op->run_func = op_default_mul_run<int64_t>;
        break;
    default:
        CHECK(false);
        break;
    }
}
