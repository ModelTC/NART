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

#include "conv_2d.h"

op_conv_2d_t *op_quant_conv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_conv_2d_t *res = (op_conv_2d_t *)malloc(sizeof(op_conv_2d_t));
    memset(res, 0, sizeof(op_conv_2d_t));
    return res;
}

void op_quant_conv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_conv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_conv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_conv_2d_t *)op)->kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_conv_2d_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_conv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_conv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_conv_2d_t *)op)->stride_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_conv_2d_t *)op)->group));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->wzero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->obits));
}

void op_quant_conv_2d_tp_destroy(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    if (NULL != conv_op->aux_mem) {
        conv_op->aux_mem = NULL;
        mem_delete(conv_op->aux_mem);
    }
    if (NULL != conv_op->temp_data) {
        mem_delete(conv_op->temp_data);
        conv_op->temp_data = NULL;
    }
    if (NULL != conv_op->tmp_output) {
        mem_delete(conv_op->tmp_output);
        conv_op->tmp_output = NULL;
    }
}

void op_quant_conv_2d_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void op_quant_conv_2d_tp_prepare(op_t *op)
{
    int i;
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    if (NULL == conv_op->temp_data)
        conv_op->temp_data = mem_new(cpu_mem_tp);
    if (NULL == conv_op->tmp_output)
        conv_op->tmp_output = mem_new(cpu_mem_tp);
    if (NULL == conv_op->aux_mem)
        conv_op->aux_mem = mem_new(cpu_mem_tp);

    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        op->run_func = op_conv_2d_u8xu8_run;
        conv_op->ugemm = gemm_i16xi16;
        conv_op->gemm_auxmem_size = gemm_i16xi16_auxmem_size;
        op_conv_2d_u8xu8_alloc_aux_mem(op);
        break;
    case dtINT8:
        if (2 >= *(conv_op->ibits) && 2 >= *(conv_op->wbits)) {
            conv_op->gemm = gemm_i2xi2;
            conv_op->gemm_auxmem_size = gemm_i2xi2_auxmem_size;
        } else if (3 >= *(conv_op->ibits) && 3 >= *(conv_op->wbits)) {
            conv_op->gemm = gemm_i3xi3;
            conv_op->gemm_auxmem_size = gemm_i3xi3_auxmem_size;
        } else if (4 >= *(conv_op->ibits) && 4 >= *(conv_op->wbits)) {
            conv_op->gemm = gemm_i4xi4;
            conv_op->gemm_auxmem_size = gemm_i4xi4_auxmem_size;
        } else if (6 >= *(conv_op->ibits) && 6 >= *(conv_op->wbits)) {
            conv_op->gemm = gemm_i6xi6;
            conv_op->gemm_auxmem_size = gemm_i6xi6_auxmem_size;
        } else if (7 >= *(conv_op->ibits) && 7 >= *(conv_op->wbits)) {
            conv_op->gemm = gemm_i7xi7;
            conv_op->gemm_auxmem_size = gemm_i7xi7_auxmem_size;
        } else {
            conv_op->gemm = gemm_i8xi8;
            conv_op->gemm_auxmem_size = gemm_i8xi8_auxmem_size;
        }
        op->run_func = op_conv_2d_i8xi8_run;
        op_conv_2d_i8xi8_alloc_aux_mem(op);
        break;
    default:
        CHECK(false);
        break;
    }
}
