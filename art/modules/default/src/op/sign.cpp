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

#include "art/data_type.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    op_t o;
} op_sign_t;
op_sign_t *op_default_sign_tp_alloc(workspace_t *ws);
void op_default_sign_tp_config(op_t *op);
void op_default_sign_tp_prepare(op_t *op);
void op_default_sign_tp_destroy(op_t *op);
void op_default_sign_tp_dealloc(op_t *op);

#ifdef __cplusplus
}
#endif

op_sign_t *op_default_sign_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_sign_t *res = (op_sign_t *)malloc(sizeof(op_sign_t));
    memset(res, 0, sizeof(op_sign_t));
    return res;
}

void op_default_sign_tp_config(op_t *op) { (void)op; }

void op_default_sign_tp_destroy(op_t *op) { (void)op; }

void op_default_sign_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T> static void op_default_sign_run_integer(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const T *input_0 = (const T *)mem_cpu_data(op->input_tensors[0]->mem);
    T *output_0 = (T *)mem_cpu_data(op->output_tensors[0].mem);
    for (size_t i = 0; i < count; ++i) {
        if (input_0[i] < 0) {
            output_0[i] = -1;
        } else if (input_0[i] > 0) {
            output_0[i] = 1;
        } else {
            output_0[i] = 0;
        }
    }
}

template <typename T> static void op_default_sign_run_float(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const T *input_0 = (const T *)mem_cpu_data(op->input_tensors[0]->mem);
    T *output_0 = (T *)mem_cpu_data(op->output_tensors[0].mem);
    float eps = 1e-9;
    for (size_t i = 0; i < count; ++i) {
        if (input_0[i] < -eps) {
            output_0[i] = -1;
        } else if (input_0[i] > eps) {
            output_0[i] = 1;
        } else {
            output_0[i] = 0;
        }
    }
}

void op_default_sign_tp_prepare(op_t *op)
{
    op_sign_t *sign_op = (op_sign_t *)op;
    tensor_alloc(op->input_tensors[0]);
    uint32_t i_dtype = op->input_tensors[0]->dtype;
    tensor_alloc(&op->output_tensors[0]);

    if (i_dtype == dtFLOAT32) {
        op->run_func = op_default_sign_run_float<float>;
    } else if (i_dtype == dtINT64) {
        op->run_func = op_default_sign_run_integer<int64_t>;
    } else {
        CHECK(false);
    }
}
