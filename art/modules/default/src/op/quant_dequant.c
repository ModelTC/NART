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
#include "art/op_settings.h"
#include "art/op_tp.h"

typedef struct {
    op_t o;
    int *q_min;
    int *q_max;
    float *scale;
} op_quant_dequant_t;

op_quant_dequant_t *op_default_quant_dequant_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_quant_dequant_t *res = (op_quant_dequant_t *)malloc(sizeof(op_quant_dequant_t));
    memset(res, 0, sizeof(op_quant_dequant_t));
    return res;
}

void op_default_quant_dequant_tp_config(op_t *op)
{
    size_t len_q_min;
    size_t len_q_max;
    size_t len_scale;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32, &len_scale,
        &((op_quant_dequant_t *)op)->scale));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_QMIN, dtINT32, &len_q_min, &((op_quant_dequant_t *)op)->q_min));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_QMAX, dtINT32, &len_q_max, &((op_quant_dequant_t *)op)->q_max));
}

void op_default_quant_dequant_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void op_default_quant_dequant_tp_destroy(op_t *op) { (void)op; }

static void op_default_quant_dequant_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const float *input = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output = (float *)mem_cpu_data(op->output_tensors[0].mem);
    size_t i;
    op_quant_dequant_t *quant_dequant_op = (op_quant_dequant_t *)op;
    size_t num_c;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32, &num_c, &quant_dequant_op->scale));
    size_t features = count / num_c;
    for (i = 0; i < count; ++i) {
        size_t c = i / features;
        output[i] = (int)round(input[i] / quant_dequant_op->scale[c]);
        output[i]
            = output[i] <= quant_dequant_op->q_max[c] ? output[i] : quant_dequant_op->q_max[c];
        output[i]
            = output[i] >= quant_dequant_op->q_min[c] ? output[i] : quant_dequant_op->q_min[c];
        output[i] *= quant_dequant_op->scale[c];
    }
}

void op_default_quant_dequant_tp_prepare(op_t *op)
{
    size_t i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[1]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_quant_dequant_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
