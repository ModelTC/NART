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
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"
#include "art/quant/quant_op_tp.h"
#include "art/timer.h"

#include "../quant_workspace.h"

typedef struct {
    op_t o;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
    int qtype;
} op_quantize_t;

static bool op_infer_shape_quant_quantize(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op_quantize_t *quantize = (op_quantize_t *)op;
        int32_t qtype = quantize->qtype << 8 | quantize->obits[0];
        switch (qtype) {
        case 0x0008:
        case 0x0108:
            op->output_tensors[0].dtype = dtUINT8;
            break;
        case 0x0202:
        case 0x0203:
        case 0x0204:
        case 0x0206:
        case 0x0207:
        case 0x0208:
            op->output_tensors[0].dtype = dtINT8;
            break;

        case 0x0010:
        case 0x0110:
            op->output_tensors[0].dtype = dtUINT16;
            break;
        case 0x0210:
            op->output_tensors[0].dtype = dtINT16;
            break;

        case 0x0020:
        case 0x0220:
            op->output_tensors[0].dtype = dtINT32;
            break;
        case 0x0120:
            op->output_tensors[0].dtype = dtUINT32;
            break;
        }
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;

    return true;
}

const op_tp_t op_quant_quantize_tp = {
    .op_tp_code = OP_QUANT_QUANTIZE,
    .name = "quantize",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_quant_quantize,
    .constraints
    = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_OALPHA, dtFLOAT32),
       OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_OZERO_POINT, dtUINT8),
       OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_OBITS, dtUINT8),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_QUANT_QTYPE, dtINT32, SETTING_QUANT_DEFAULT),
       OP_SETTING_CONSTRAINT_END()}
};

/*
alpha = (max - min) / (2 ^ bits - 1)
beta = min
quant = (real - beta) / scale
zero_point = (0 - beta) / scale
real = (quant - zero_point) * scale
*/

op_quantize_t *op_quant_quantize_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_quantize_t *res = (op_quantize_t *)malloc(sizeof(op_quantize_t));
    memset(res, 0, sizeof(op_quantize_t));
    return res;
}

void op_quant_quantize_tp_config(op_t *op)
{
    size_t len_oalpha;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_oalpha, &((op_quantize_t *)op)->oalpha));

    size_t len_ozero_point;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_ozero_point,
        &((op_quantize_t *)op)->ozero_point));

    size_t len_bits;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_quantize_t *)op)->obits));
    CHECK(op_setting_single_get(op, SETTING_QUANT_QTYPE, dtINT32, &((op_quantize_t *)op)->qtype));
}

void op_quant_quantize_tp_destroy(op_t *op) { (void)op; }

void op_quant_quantize_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quantize_run_u8(op_t *op)
{
    size_t num = shape_count(&op->input_tensors[0]->shape);
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    op_quantize_t *quantize_op = (op_quantize_t *)op;
    float alpha = quantize_op->oalpha[0];
    float beta = -(quantize_op->oalpha[0] * quantize_op->ozero_point[0]);
    size_t i;
    for (i = 0; i < num; i++) {
        output_0[i] = saturate_float_by_bits((input_0[i] - beta) / alpha, quantize_op->obits[0]);
    }
}

static void op_quantize_run_i8(op_t *op)
{
    size_t num = shape_count(&op->input_tensors[0]->shape);
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    int8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    op_quantize_t *quantize_op = (op_quantize_t *)op;
    float alpha = 1. / quantize_op->oalpha[0];
    size_t i;
    int l = (1l << (quantize_op->obits[0] - 1)) - 1;
    for (i = 0; i < num / 8; i++) {
        output_0[0] = clip(round(input_0[0] * alpha), -l, l);
        output_0[1] = clip(round(input_0[1] * alpha), -l, l);
        output_0[2] = clip(round(input_0[2] * alpha), -l, l);
        output_0[3] = clip(round(input_0[3] * alpha), -l, l);
        output_0[4] = clip(round(input_0[4] * alpha), -l, l);
        output_0[5] = clip(round(input_0[5] * alpha), -l, l);
        output_0[6] = clip(round(input_0[6] * alpha), -l, l);
        output_0[7] = clip(round(input_0[7] * alpha), -l, l);
        input_0 += 8;
        output_0 += 8;
    }
    for (i = 0; i < num % 8; i++) {
        output_0[0] = clip(round(input_0[0] * alpha), -l, l);
        input_0++;
        output_0++;
    }
}

static void op_quantize_run_u16(op_t *op)
{
    size_t num = shape_count(&op->input_tensors[0]->shape);
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint16_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    op_quantize_t *quantize_op = (op_quantize_t *)op;
    float alpha = quantize_op->oalpha[0];
    float beta = -(quantize_op->oalpha[0] * quantize_op->ozero_point[0]);
    size_t i;
    for (i = 0; i < num; i++) {
        output_0[i] = saturate_float_by_bits((input_0[i] - beta) / alpha, quantize_op->obits[0]);
    }
}

static void op_quantize_run_i32(op_t *op)
{
    size_t num = shape_count(&op->input_tensors[0]->shape);
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    int32_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    op_quantize_t *quantize_op = (op_quantize_t *)op;
    float alpha = quantize_op->oalpha[0];
    float beta = -(quantize_op->oalpha[0] * quantize_op->ozero_point[0]);
    size_t i;
    for (i = 0; i < num; i++) {
        output_0[i] = (input_0[i] - beta) / alpha;
    }
}

void op_quant_quantize_tp_prepare(op_t *op)
{
    const op_quantize_t *quantize = (op_quantize_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        CHECK(op->input_tensors[i]->dtype == dtFLOAT32);
        if (quantize->qtype == SETTING_QUANT_SYMMETRIC && quantize->ozero_point[i] != 0) {
            LOG_warn(
                "ZERO_POINT will not be used when set SYMMETRIC for tensor [%s].\n",
                op->input_tensors[i]->name);
        }
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    int32_t qtype = quantize->qtype << 8 | quantize->obits[0];

    switch (qtype) {
    case 0x0008:
    case 0x0108:
        op->run_func = op_quantize_run_u8;
        break;
    case 0x0202:
    case 0x0203:
    case 0x0204:
    case 0x0206:
    case 0x0207:
    case 0x0208:
        op->run_func = op_quantize_run_i8;
        break;

    case 0x0010:
    case 0x0110:
        op->run_func = op_quantize_run_u16;
        break;

    case 0x0020:
    case 0x0220:
        op->run_func = op_quantize_run_i32;
        break;

    default:
        LOG_error(
            "Quantization for %d-bits using type[%d] is not implemented.\n", quantize->obits[0],
            quantize->qtype);
        CHECK(false);
    }
}
