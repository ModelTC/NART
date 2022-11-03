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

#include <math.h>
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
} op_div_t;
op_div_t *op_default_div_tp_alloc(workspace_t *ws);
void op_default_div_tp_config(op_t *op);
void op_default_div_tp_destroy(op_t *op);
void op_default_div_tp_dealloc(op_t *op);
void op_default_div_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif

op_div_t *op_default_div_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_div_t *res = (op_div_t *)malloc(sizeof(op_div_t));
    memset(res, 0, sizeof(op_div_t));
    return res;
}

void op_default_div_tp_config(op_t *op) { (void)op; }

void op_default_div_tp_destroy(op_t *op) { (void)op; }

void op_default_div_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

// Trunc Div
template <typename T> void _div(const T *D, const T *d, T *q) { (*q) = (*D) / (*d); }

// Floor Div
template <> void _div<int32_t>(const int32_t *D, const int32_t *d, int32_t *q)
{
    *q = (*D) / (*d);
    int32_t r = (*D) % (*d);
    if ((r > 0 && (*d) < 0) || (r < 0 && (*d) > 0))
        (*q) -= 1;
}

template <typename T> static void op_default_div_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    int i, j;
    const T *input_0 = (const T *)mem_cpu_data(op->input_tensors[0]->mem);
    const T *input_1 = (const T *)mem_cpu_data(op->input_tensors[1]->mem);
    T *output_0 = (T *)mem_cpu_data(op->output_tensors[0].mem);
    if (tensor_shape_eq(&op->input_tensors[0]->shape, &op->input_tensors[1]->shape)) {
        for (i = 0; i < (int)count; ++i) {
            _div(input_0 + i, input_1 + i, output_0 + i);
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
        if (shape_part_count(&op->input_tensors[1]->shape, 1) == 1) {
            T t = input_1[0];
            for (i = 0; i < (int)count - 4; i = i + 4) {
                _div(input_0, &t, output_0);
                _div(input_0 + 1, &t, output_0 + 1);
                _div(input_0 + 2, &t, output_0 + 2);
                _div(input_0 + 3, &t, output_0 + 3);
                output_0 += 4;
                input_0 += 4;
            }
            for (; i < (int)count; ++i) {
                _div(input_0++, &t, output_0++);
            }
        } else
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
                _div(input_0 + id1, input_1 + id2, output_0 + i);
            }
    }
}

static void op_default_div_run_old(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    int i, j;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    if (tensor_shape_eq(&op->input_tensors[0]->shape, &op->input_tensors[1]->shape)) {
        for (i = 0; i < (int)count; ++i) {
            output_0[i] = input_0[i] / input_1[i];
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
        if (shape_part_count(&op->input_tensors[1]->shape, 1) == 1) {
            float t = input_1[0];
            for (i = 0; i < (int)count - 4; i = i + 4) {
                output_0[0] = input_0[0] / t;
                output_0[0 + 1] = input_0[0 + 1] / t;
                output_0[0 + 2] = input_0[0 + 2] / t;
                output_0[0 + 3] = input_0[0 + 3] / t;
                output_0 += 4;
                input_0 += 4;
            }
            for (; i < (int)count; ++i) {
                *output_0++ = *input_0++ / t;
            }
        } else
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
                output_0[i] = input_0[id1] / input_1[id2];
            }
    }
}
void op_default_div_tp_prepare(op_t *op)
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
        op->run_func = op_default_div_run<float>;
        //            op->run_func = op_default_div_run_old;
        break;
    case dtINT32:
        op->run_func = op_default_div_run<int32_t>;
        break;
    default:
        CHECK(false);
        break;
    }
}
