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

typedef struct {
    op_t o;
} op_exchange_t;

op_exchange_t *op_default_exchange_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_exchange_t *res = (op_exchange_t *)malloc(sizeof(op_exchange_t));
    memset(res, 0, sizeof(op_exchange_t));
    return res;
}

void op_default_exchange_tp_config(op_t *op) { (void)op; }

void op_default_exchange_tp_destroy(op_t *op) { (void)op; }

void op_default_exchange_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_exchange_run(op_t *op)
{
    size_t i, j;
    size_t width_in = op->input_tensors[0]->shape.dim[3];
    size_t nch_in = op->input_tensors[0]->shape.dim[0] * op->input_tensors[0]->shape.dim[1]
        * op->input_tensors[0]->shape.dim[2];
    size_t count = shape_count(&op->input_tensors[0]->shape);
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (i = 0; i < width_in; ++i) {
        for (j = 0; j < nch_in; ++j) {
            *output_0++ = *input_0;
            input_0 += width_in;
        }
        input_0 -= count - 1;
    }
}

void op_default_exchange_tp_prepare(op_t *op)
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
        op->run_func = op_default_exchange_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
