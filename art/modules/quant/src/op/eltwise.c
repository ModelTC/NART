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
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"

#undef max // msvc defined a macro called max

typedef struct {
    op_t o;
    uint32_t operation;
    float *coeff;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    uint32_t *multer;
    int8_t *shift;
    int8_t shift0;

    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_eltwise_t;

op_eltwise_t *op_quant_eltwise_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_eltwise_t *res = (op_eltwise_t *)malloc(sizeof(op_eltwise_t));
    memset(res, 0, sizeof(op_eltwise_t));
    return res;
}

void op_quant_eltwise_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ELTWISE_OPERATION, dtUINT32, &((op_eltwise_t *)op)->operation));
    if (!op_setting_if_set(op, SETTING_ELTWISE_COEFF)) {
        float *coeff = (float *)malloc(sizeof(float) * op->input_size);
        int i;
        for (i = 0; i < op->input_size; ++i) {
            coeff[i] = 1.0;
        }
        CHECK(op_setting_array_set(op, SETTING_ELTWISE_COEFF, dtFLOAT32, op->input_size, coeff));
        free(coeff);
    }
    size_t len;
    CHECK(op_setting_array_get(
        op, SETTING_ELTWISE_COEFF, dtFLOAT32, &len, &((op_eltwise_t *)op)->coeff));
    CHECK_EQ(op->input_size, len);

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_eltwise_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_eltwise_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_eltwise_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_eltwise_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_eltwise_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_eltwise_t *)op)->obits));
}

void op_quant_eltwise_tp_destroy(op_t *op)
{
    op_eltwise_t *eltwise = (op_eltwise_t *)op;
    if (NULL != eltwise->multer) {
        free(eltwise->multer);
    }
    if (NULL != eltwise->shift) {
        free(eltwise->shift);
    }
}

void op_quant_eltwise_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_eltwise_fill_output(op_t *op, float k)
{
    uint32_t i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (i = 0; i < count; ++i)
        output[i] = k;
}

static void op_quant_eltwise_accumulate_output_prod(op_t *op)
{
    op_quant_eltwise_fill_output(op, 1.0);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < op->input_size; ++n) {
        const float *input = mem_cpu_data(op->input_tensors[n]->mem);
        for (i = 0; i < count; ++i)
            output[i] *= input[i];
    }
}

static void op_quant_eltwise_accumulate_output_sum(op_t *op)
{
    // op_quant_eltwise_fill_output(op, 0.0);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    uint8_t *output = mem_cpu_data(op->output_tensors[0].mem);

    /*
    for (n = 0; n < op->input_size; ++n) {
    const float *input = mem_cpu_data(op->input_tensors[n]->mem);
    float c = coeff ? coeff[n] : 1.0;
    for (i = 0; i < count; ++i)
    output[i] += input[i] * c;
    }
    */
    uint32_t *multer = ((op_eltwise_t *)op)->multer;
    uint8_t *izero_point = ((op_eltwise_t *)op)->izero_point;
    uint8_t *ozero_point = ((op_eltwise_t *)op)->ozero_point;
    int8_t *shift = ((op_eltwise_t *)op)->shift;
    int8_t shift0 = ((op_eltwise_t *)op)->shift0 - 1;

    uint8_t **inputs = (uint8_t **)malloc(sizeof(uint8_t *) * op->input_size);
    for (n = 0; n < op->input_size; ++n) {
        inputs[n] = mem_cpu_data(op->input_tensors[n]->mem);
    }

    uint8_t bit = ((op_eltwise_t *)op)->obits[0];
    for (i = 0; i < count; ++i) {
        int32_t tmp = 0;
        for (n = 0; n < op->input_size; ++n) {
            tmp += rshift_rn(
                (int32_t)(inputs[n][i] - izero_point[n]) * (int32_t)multer[n], shift[n] - shift0);
        }
        output[i] = saturate_int_by_bits(rshift_rn(tmp, shift0) + ozero_point[0], bit);
    }
    free(inputs);
}

static int max(int a, int b) { return a < b ? b : a; }

static void op_quant_eltwise_accumulate_output_max(op_t *op)
{
    // op_quant_eltwise_fill_output(op, -FLT_MAX);

    uint32_t n, i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    uint8_t *output = mem_cpu_data(op->output_tensors[0].mem);

    uint32_t *multer = ((op_eltwise_t *)op)->multer;
    uint8_t *izero_point = ((op_eltwise_t *)op)->izero_point;
    uint8_t *ozero_point = ((op_eltwise_t *)op)->ozero_point;
    int8_t *shift = ((op_eltwise_t *)op)->shift;
    int8_t shift0 = ((op_eltwise_t *)op)->shift0 - 1;

    uint8_t **inputs = (uint8_t **)malloc(sizeof(uint8_t *) * op->input_size);
    for (n = 0; n < op->input_size; ++n) {
        inputs[n] = mem_cpu_data(op->input_tensors[n]->mem);
    }

    uint8_t bit = ((op_eltwise_t *)op)->obits[0];
    for (i = 0; i < count; ++i) {
        int32_t tmp = 0xffffffff;
        for (n = 0; n < op->input_size; ++n) {
            tmp = max(
                tmp,
                rshift_rn(
                    (int32_t)(inputs[n][i] - izero_point[n]) * (int32_t)multer[n],
                    shift[n] - shift0));
        }
        output[i] = saturate_int_by_bits(rshift_rn(tmp, shift0) + ozero_point[0], bit);
    }
    free(inputs);
}

void op_quant_eltwise_tp_prepare(op_t *op)
{
    CHECK(
        ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_PROD
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_SUM
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_MAX);

    op_eltwise_t *eltwise = (op_eltwise_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        break;
    default:
        CHECK(false);
        break;
    }
    if (NULL != op->run_func)
        return;
    switch (((op_eltwise_t *)op)->operation) {
    case SETTING_ELTWISE_OP_PROD:
        CHECK(false);
        op->run_func = op_quant_eltwise_accumulate_output_prod;
        break;
    case SETTING_ELTWISE_OP_SUM:
        eltwise->multer = (uint32_t *)malloc(sizeof(uint32_t) * op->input_size);
        eltwise->shift = (int8_t *)malloc(sizeof(int8_t) * op->input_size);
        eltwise->shift0 = 127;
        for (i = 0; i < op->input_size; ++i) {
            get_quant_scale(
                eltwise->coeff[i] * eltwise->ialpha[i] / eltwise->oalpha[0], &eltwise->multer[i],
                &eltwise->shift[i], 23);
            eltwise->shift0
                = eltwise->shift0 > eltwise->shift[i] ? eltwise->shift[i] : eltwise->shift0;
        }
        op->run_func = op_quant_eltwise_accumulate_output_sum;
        break;
    case SETTING_ELTWISE_OP_MAX:
        eltwise->multer = (uint32_t *)malloc(sizeof(uint32_t) * op->input_size);
        eltwise->shift = (int8_t *)malloc(sizeof(int8_t) * op->input_size);
        eltwise->shift0 = 127;
        for (i = 0; i < op->input_size; ++i) {
            get_quant_scale(
                eltwise->coeff[i] * eltwise->ialpha[i] / eltwise->oalpha[0], &eltwise->multer[i],
                &eltwise->shift[i], 23);
            eltwise->shift0
                = eltwise->shift0 > eltwise->shift[i] ? eltwise->shift[i] : eltwise->shift0;
        }
        op->run_func = op_quant_eltwise_accumulate_output_max;
        break;
    default:
        CHECK(false);
        break;
    }
}
