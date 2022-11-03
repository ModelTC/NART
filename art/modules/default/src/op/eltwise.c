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

#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

typedef struct {
    op_t o;
    uint32_t operation;
    float *coeff;
} op_eltwise_t;

op_eltwise_t *op_default_eltwise_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_eltwise_t *res = (op_eltwise_t *)malloc(sizeof(op_eltwise_t));
    memset(res, 0, sizeof(op_eltwise_t));
    return res;
}

void op_default_eltwise_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ELTWISE_OPERATION, dtUINT32, &((op_eltwise_t *)op)->operation));
    if (op_setting_if_set(op, SETTING_ELTWISE_COEFF)) {
        size_t len;
        CHECK(op_setting_array_get(
            op, SETTING_ELTWISE_COEFF, dtFLOAT32, &len, &((op_eltwise_t *)op)->coeff));
        CHECK_EQ(op->input_size, len);
    }
}

void op_default_eltwise_tp_destroy(op_t *op) { (void)op; }

void op_default_eltwise_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_eltwise_fill_output(op_t *op, float k)
{
    uint32_t i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (i = 0; i < count; ++i)
        output[i] = k;
}

static void op_default_eltwise_accumulate_output_prod(op_t *op)
{
    op_default_eltwise_fill_output(op, 1.0);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < op->input_size; ++n) {
        const float *input = mem_cpu_data(op->input_tensors[n]->mem);
        for (i = 0; i < count; ++i)
            output[i] *= input[i];
    }
}

static void op_default_eltwise_accumulate_output_sum(op_t *op)
{
    op_default_eltwise_fill_output(op, 0.0);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    const float *coeff = ((op_eltwise_t *)op)->coeff;
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < op->input_size; ++n) {
        const float *input = mem_cpu_data(op->input_tensors[n]->mem);
        float c = coeff ? coeff[n] : 1.0;
        for (i = 0; i < count; ++i)
            output[i] += input[i] * c;
    }
}

static void op_default_eltwise_accumulate_output_max(op_t *op)
{
    op_default_eltwise_fill_output(op, -FLT_MAX);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < op->input_size; ++n) {
        const float *input = mem_cpu_data(op->input_tensors[n]->mem);
        for (i = 0; i < count; ++i)
            if (output[i] < input[i])
                output[i] = input[i];
    }
}

void op_default_eltwise_tp_prepare(op_t *op)
{
    CHECK(
        ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_PROD
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_SUM
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_MAX);

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        break;
    default:
        CHECK(false);
        break;
    }
    switch (((op_eltwise_t *)op)->operation) {
    case SETTING_ELTWISE_OP_PROD:
        op->run_func = op_default_eltwise_accumulate_output_prod;
        break;
    case SETTING_ELTWISE_OP_SUM:
        op->run_func = op_default_eltwise_accumulate_output_sum;
        break;
    case SETTING_ELTWISE_OP_MAX:
        op->run_func = op_default_eltwise_accumulate_output_max;
        break;
    default:
        CHECK(false);
        break;
    }
}
