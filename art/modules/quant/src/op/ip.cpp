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
#include "art/tensor.h"

#include "../gemm/gemm.h"
#include "../quant_workspace.h"
#include "../utils/im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;

    float *walpha;
    uint8_t *wzero_point;
    uint8_t *wbits;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

    void (*gemm)(
        const size_t, const size_t, const size_t, const int8_t *, const int8_t *, int32_t *,
        mem_t *);
    void (*ugemm)(
        const size_t, const size_t, const size_t, const int16_t *, const int16_t *, int32_t *,
        mem_t *);
    size_t (*gemm_auxmem_size)(int, int, int);

    mem_t *aux_mem;
    mem_t *tmp_output;
    mem_t *temp_data;
} op_ip_t;
op_ip_t *op_quant_ip_tp_alloc(workspace_t *ws);
void op_quant_ip_tp_config(op_t *op);
void op_quant_ip_tp_destroy(op_t *op);
void op_quant_ip_tp_dealloc(op_t *op);
void op_quant_ip_tp_prepare(op_t *op);
#ifdef __cplusplus
}
#endif
op_ip_t *op_quant_ip_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_ip_t *res = (op_ip_t *)malloc(sizeof(op_ip_t));
    memset(res, 0, sizeof(op_ip_t));
    return res;
}

void op_quant_ip_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_IP_NUM_OUTPUT, dtUINT32, &((op_ip_t *)op)->num_output));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_ip_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point, &((op_ip_t *)op)->izero_point));
    CHECK(
        op_setting_array_get(op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_ip_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_ip_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WZERO_POINT, dtUINT8, &len_zero_point, &((op_ip_t *)op)->wzero_point));
    CHECK(
        op_setting_array_get(op, SETTING_QUANT_WBITS, dtUINT8, &len_bits, &((op_ip_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_ip_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point, &((op_ip_t *)op)->ozero_point));
    CHECK(
        op_setting_array_get(op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_ip_t *)op)->obits));
}

void op_quant_ip_tp_destroy(op_t *op)
{
    op_ip_t *ip_op = (op_ip_t *)op;
    if (NULL != ip_op->aux_mem) {
        mem_delete(ip_op->aux_mem);
        ip_op->aux_mem = NULL;
    }
    if (NULL != ip_op->temp_data) {
        mem_delete(ip_op->temp_data);
        ip_op->temp_data = NULL;
    }
    if (NULL != ip_op->tmp_output) {
        mem_delete(ip_op->tmp_output);
        ip_op->tmp_output = NULL;
    }
}

void op_quant_ip_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_ip_u8xu8_run(op_t *op)
{
    size_t i, j;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t i_chw = shape_part_count(&(op->input_tensors[0]->shape), 1);

    const op_ip_t *ip_op = (op_ip_t *)op;
    const uint8_t *input_0 = (const uint8_t *)mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *input_1 = (const uint8_t *)mem_cpu_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = NULL;
    uint8_t *output_0 = (uint8_t *)mem_cpu_data(op->output_tensors[0].mem);

    if (op->input_size == 3)
        input_2 = (const int32_t *)mem_cpu_data(op->input_tensors[2]->mem);

    float walpha = ip_op->walpha[0];
    uint8_t wzero_point = ip_op->wzero_point[0];

    float ialpha = ip_op->ialpha[0];
    uint8_t izero_point = ip_op->izero_point[0];

    float oalpha = ip_op->oalpha[0];
    uint8_t ozero_point = ip_op->ozero_point[0];

#ifdef USE_FIXED_POINT_ONLY
    /* alpha = result_mult / 2 ^ result_shift */
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif

    size_t input0_cnt = shape_count(&op->input_tensors[0]->shape);
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    CHECK_GE(
        mem_sizeof(ip_op->temp_data),
        shape_count(&op->input_tensors[0]->shape) * sizeof(int16_t)
            + shape_count(&op->input_tensors[1]->shape) * sizeof(int16_t));

    int16_t *tmp_input0 = (int16_t *)mem_cpu_data(ip_op->temp_data);
    int16_t *tmp_input1 = tmp_input0 + shape_count(&op->input_tensors[0]->shape);

    CHECK_GE(mem_sizeof(ip_op->tmp_output), output_cnt * sizeof(int32_t));
    int32_t *tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
    for (i = 0; i < input0_cnt; i++) {
        tmp_input0[i] = input_0[i] - izero_point;
    }

    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);
    for (i = 0; i < input1_cnt; i++) {
        tmp_input1[i] = input_1[i] - wzero_point;
    }

    if (op->input_size == 3) {
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < ip_op->num_output; ++j) {
                *tmp_output++ = input_2[j];
            }
        }
    }

    tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
    for (i = 0; i < batch_size; ++i) {
        ip_op->ugemm(
            ip_op->num_output, 1, i_chw, tmp_input1, tmp_input0, tmp_output, ip_op->aux_mem);
        tmp_input0 += i_chw;
        tmp_output += ip_op->num_output;
    }
    uint8_t bit = ip_op->obits[0];
    tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
    for (i = 0; i < output_cnt; i++) {
        *output_0++ = saturate_int_by_bits(
            rshift_rn((int64_t)(*(tmp_output)++) * (int64_t)result_mult, result_shift)
                + ozero_point,
            bit);
    }
}

static void op_quant_ip_i8xi8_run(op_t *op)
{
    size_t i, j;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t i_chw = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
        * op->input_tensors[0]->shape.dim[3];

    const int8_t *input_0 = (const int8_t *)mem_cpu_data(op->input_tensors[0]->mem);
    const int8_t *input_1 = (const int8_t *)mem_cpu_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = NULL;
    if (op->input_size == 3)
        input_2 = (const int32_t *)mem_cpu_data(op->input_tensors[2]->mem);
    int8_t *output_0 = (int8_t *)mem_cpu_data(op->output_tensors[0].mem);
    const op_ip_t *ip_op = (op_ip_t *)op;

    float walpha = ip_op->walpha[0];
    float ialpha = ip_op->ialpha[0];
    float oalpha = ip_op->oalpha[0];

#ifdef USE_FIXED_POINT_ONLY
    /* alpha = result_mult / 2 ^ result_shift */
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    CHECK_GE(mem_sizeof(ip_op->tmp_output), output_cnt * sizeof(int32_t));
    int32_t *tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
    if (op->input_size == 3) {
#ifndef __aarch64__
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < ip_op->num_output; ++j) {
                *tmp_output++ = input_2[j];
            }
        }
        tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
#endif
    } else {
        memset(tmp_output, 0, sizeof(int32_t) * output_cnt);
    }

    for (i = 0; i < batch_size; ++i) {
        ip_op->gemm(ip_op->num_output, 1, i_chw, input_1, input_0, tmp_output, ip_op->aux_mem);
        input_0 += i_chw;
        tmp_output += ip_op->num_output;
    }
    tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);

#if defined(__aarch64__) && !defined(__APPLE__)
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < ip_op->num_output; ++j) {
            *tmp_output++ += input_2[j];
        }
    }
    tmp_output = (int32_t *)mem_cpu_data(ip_op->tmp_output);
#endif

    uint8_t bit = ip_op->obits[0];
    for (i = 0; i < output_cnt; i++) {
        *output_0++ = ssaturate_int_by_bits(
            rshift_rn((int64_t)(*(tmp_output)++) * (int64_t)result_mult, result_shift), bit);
    }
}

void op_quant_ip_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    op_ip_t *ip_op = (op_ip_t *)op;
    if (NULL == ip_op->aux_mem)
        ip_op->aux_mem = mem_new(cpu_mem_tp);
    if (NULL == ip_op->tmp_output)
        ip_op->tmp_output = mem_new(cpu_mem_tp);
    if (NULL == ip_op->temp_data)
        ip_op->temp_data = mem_new(cpu_mem_tp);

    mem_alloc(ip_op->tmp_output, sizeof(int32_t) * shape_count(&op->output_tensors[0].shape));

    switch (op->input_tensors[0]->dtype) {
    case dtUINT8: {
        ip_op->ugemm = gemm_i16xi16;
        ip_op->gemm_auxmem_size = gemm_i16xi16_auxmem_size;

        mem_alloc(
            ip_op->temp_data,
            sizeof(int16_t) * shape_count(&op->input_tensors[1]->shape)
                + sizeof(int16_t) * shape_count(&op->input_tensors[0]->shape));
        int M = ip_op->num_output;
        int N = 1;
        int K = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
            * op->input_tensors[0]->shape.dim[3];
        size_t sz = MAX(M * N * sizeof(int32_t), M * K * sizeof(int16_t));
        mem_alloc(ip_op->aux_mem, sz);
        op->run_func = op_quant_ip_u8xu8_run;
    } break;
    case dtINT8: {
        if (6 == *(ip_op->wbits) && 6 == *(ip_op->ibits) && 32 == *(ip_op->obits)) {
            ip_op->gemm = gemm_i6xi6;
            ip_op->gemm_auxmem_size = gemm_i6xi6_auxmem_size;
        } else if (7 == *(ip_op->wbits) && 7 == *(ip_op->ibits) && 32 == *(ip_op->obits)) {
            ip_op->gemm = gemm_i7xi7;
            ip_op->gemm_auxmem_size = gemm_i7xi7_auxmem_size;
        } else {
            ip_op->gemm = gemm_i8xi8;
            ip_op->gemm_auxmem_size = gemm_i8xi8_auxmem_size;
        }

        int M = ip_op->num_output;
        int N = 1;
        int K = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
            * op->input_tensors[0]->shape.dim[3];
        size_t sz = ip_op->gemm_auxmem_size(M, N, K);
        mem_alloc(ip_op->aux_mem, sz);
        op->run_func = op_quant_ip_i8xi8_run;
    } break;
    default:
        CHECK(false);
        break;
    }
}
