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

typedef struct {
    op_t o;
} op_sigmoid_t;

op_sigmoid_t *op_default_sigmoid_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_sigmoid_t *res = (op_sigmoid_t *)malloc(sizeof(op_sigmoid_t));
    memset(res, 0, sizeof(op_sigmoid_t));
    return res;
}

void op_default_sigmoid_tp_config(op_t *op) { (void)op; }

void op_default_sigmoid_tp_destroy(op_t *op) { (void)op; }

void op_default_sigmoid_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_sigmoid_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    size_t i;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    for (i = 0; i < count; ++i) {
        output_0[i] = 1.0 / (1.0 + exp(-input_0[i]));
    }
}

void op_default_sigmoid_tp_prepare(op_t *op)
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
        op->run_func = op_default_sigmoid_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
