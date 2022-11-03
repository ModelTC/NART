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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/cuda_quant/cuda_quant_mem.h"
#include "art/cuda_quant/cuda_quant_op_settings.h"
#include "art/cuda_quant/cuda_quant_op_tp.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_quant_workspace.h"
#include "../utils/cuda_quant_helper.cuh"

static bool op_infer_shape_quant_dequantize(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = dtFLOAT32;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;

    return true;
}

const op_tp_t op_dequantize_tp = {
    .op_tp_code = OP_CUDA_QUANT_DEQUANTIZE,
    .name = "dequantize",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_quant_dequantize,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_CUDA_QUANT_IALPHA, dtFLOAT32),
                    OP_SETTING_CONSTRAINT_REPEATED(SETTING_CUDA_QUANT_IZERO_POINT, dtUINT8),
                    OP_SETTING_CONSTRAINT_REPEATED(SETTING_CUDA_QUANT_IBITS, dtUINT8),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CUDA_QUANT_QTYPE, dtINT32, 0),
                    OP_SETTING_CONSTRAINT_END()}
};

/*
alpha = (max - min) / (2 ^ bits - 1)
beta = min
quant = (real - beta) / scale
zero_point = (0 - beta) / scale
real = (quant - zero_point) * scale
*/

typedef struct {
    op_t o;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    int qtype;
} op_cuda_dequantize_t;

op_cuda_dequantize_t *op_cuda_quant_dequantize_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_cuda_dequantize_t *res = (op_cuda_dequantize_t *)malloc(sizeof(op_cuda_dequantize_t));
    memset(res, 0, sizeof(op_cuda_dequantize_t));
    return res;
}

void op_cuda_quant_dequantize_tp_config(op_t *op)
{
    size_t len_alpha;
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IALPHA, dtFLOAT32, &len_alpha,
        &((op_cuda_dequantize_t *)op)->ialpha));

    size_t len_zero_point;
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_cuda_dequantize_t *)op)->izero_point));

    size_t len_bits;
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IBITS, dtUINT8, &len_bits, &((op_cuda_dequantize_t *)op)->ibits));
    CHECK(op_setting_single_get(
        op, SETTING_CUDA_QUANT_QTYPE, dtINT32, &((op_cuda_dequantize_t *)op)->qtype));
}

void op_cuda_quant_dequantize_tp_destroy(op_t *op) { (void)op; }

void op_cuda_quant_dequantize_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_cuda_dequantize_run_u8(op_t *op)
{
    (void)op;
    /*
    size_t num = shape_count(&op->input_tensors[0]->shape);
    const uint8_t* input_0 = mem_data(op->input_tensors[0]->mem);
    float* output_0 = mem_data(op->output_tensors[0].mem);
    op_cuda_dequantize_t* dequantize_op = (op_cuda_dequantize_t*)op;
    float alpha = dequantize_op->ialpha[0];
    uint8_t zero_point = dequantize_op->izero_point[0];
    dequantize_u8_cu(input_0, output_0, num, zero_point, alpha, op->workspace);
    */
}

static void op_cuda_dequantize_run_i8(op_t *op)
{
    // size_t num = shape_count(&op->input_tensors[0]->shape);
    const int8_t *input_0 = mem_data(op->input_tensors[0]->mem);
    float *output_0 = mem_data(op->output_tensors[0].mem);
    op_cuda_dequantize_t *dequantize_op = (op_cuda_dequantize_t *)op;
    float alpha = dequantize_op->ialpha[0];
    shape_t *in_shape = &op->input_tensors[0]->shape;
    int n = in_shape->dim[0];
    int c = in_shape->dim[1];
    int h = in_shape->dim[2];
    int w = in_shape->dim[3];

    // default transpose
    // dequantize_i8_cu(input_0, output_0, num, alpha, op->workspace);
    transpose_NHWC_2_NCHW_dequantize_i8_cu(input_0, output_0, n, c, h, w, alpha, op->workspace);
}

static void op_cuda_dequantize_run_i4(op_t *op)
{
    const int8_t *input_0 = mem_data(op->input_tensors[0]->mem);
    float *output_0 = mem_data(op->output_tensors[0].mem);
    op_cuda_dequantize_t *dequantize_op = (op_cuda_dequantize_t *)op;
    float alpha = dequantize_op->ialpha[0];
    shape_t *in_shape = &op->input_tensors[0]->shape;
    int n = in_shape->dim[0];
    int c = in_shape->dim[1];
    int h = in_shape->dim[2];
    int w = in_shape->dim[3];

    // default transpose
    // dequantize_i4_cu(input_0, output_0, num, alpha, op->workspace);
    transpose_NHWC_2_NCHW_dequantize_i4_cu(input_0, output_0, n, c, h, w, alpha, op->workspace);
}

void op_cuda_quant_dequantize_tp_prepare(op_t *op)
{
    const op_cuda_dequantize_t *dequantize = (op_cuda_dequantize_t *)op;
    int32_t qtype = dequantize->qtype << 8 | dequantize->ibits[0];
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (qtype) {
    case 0x0008:
    case 0x0108:
        op->run_func = op_cuda_dequantize_run_u8;
        break;
    case 0x0202:
    case 0x0203:
    case 0x0204:
        op->run_func = op_cuda_dequantize_run_i4;
        break;
    case 0x0206:
    case 0x0207:
    case 0x0208:
        op->run_func = op_cuda_dequantize_run_i8;
        break;

    default:
        LOG_error(
            "Dequantization for %d-bits using type[%d] is not implemented.\n", dequantize->ibits[0],
            dequantize->qtype);
        CHECK(false);
    }
}
