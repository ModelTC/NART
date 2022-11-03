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

#include "conv_2d_wino.h"

op_conv_2d_wino_t *op_quant_conv_2d_wino_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_conv_2d_wino_t *res = ((op_conv_2d_wino_t *)malloc(sizeof(op_conv_2d_wino_t)));
    memset(res, 0, sizeof(op_conv_2d_wino_t));
    return res;
}

void op_quant_conv_2d_wino_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_conv_2d_wino_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_conv_2d_wino_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_conv_2d_wino_t *)op)->kernel_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_conv_2d_wino_t *)op)->pad_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_conv_2d_wino_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_conv_2d_wino_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_conv_2d_wino_t *)op)->stride_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_conv_2d_wino_t *)op)->group));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_wino_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_wino_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_conv_2d_wino_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_wino_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_wino_t *)op)->wzero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WBITS, dtUINT8, &len_bits, &((op_conv_2d_wino_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_wino_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_wino_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_conv_2d_wino_t *)op)->obits));
}

void op_quant_conv_2d_wino_tp_destroy(op_t *op)
{
    op_conv_2d_wino_t *conv_op = (op_conv_2d_wino_t *)op;
    if (NULL != conv_op->aux_mem1) {
        mem_delete(conv_op->aux_mem1);
        conv_op->aux_mem1 = NULL;
    }
    if (NULL != conv_op->aux_mem2) {
        mem_delete(conv_op->aux_mem2);
        conv_op->aux_mem2 = NULL;
    }
    if (NULL != conv_op->aux_mem3) {
        mem_delete(conv_op->aux_mem3);
        conv_op->aux_mem3 = NULL;
    }
    if (NULL != conv_op->aux_mem4) {
        mem_delete(conv_op->aux_mem4);
        conv_op->aux_mem4 = NULL;
    }
    if (NULL != conv_op->tmp_output) {
        mem_delete(conv_op->tmp_output);
        conv_op->tmp_output = NULL;
    }
}

void op_quant_conv_2d_wino_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void op_quant_conv_2d_wino_tp_prepare(op_t *op)
{
    int i;
    op_conv_2d_wino_t *conv_op = (op_conv_2d_wino_t *)op;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    if (NULL == conv_op->aux_mem1)
        conv_op->aux_mem1 = mem_new(cpu_mem_tp);
    if (NULL == conv_op->aux_mem2)
        conv_op->aux_mem2 = mem_new(cpu_mem_tp);
    if (NULL == conv_op->aux_mem3)
        conv_op->aux_mem3 = mem_new(cpu_mem_tp);
    if (NULL == conv_op->aux_mem4)
        conv_op->aux_mem4 = mem_new(cpu_mem_tp);
    if (NULL == conv_op->tmp_output)
        conv_op->tmp_output = mem_new(cpu_mem_tp);

    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
    // op_conv_2d_wino_u8xu8_alloc_aux_mem(op);
    // op->run_func = op_conv_2d_wino_u8xu8_run;
    // break;
    case dtINT8:
        op_conv_2d_wino_i8xi8_alloc_aux_mem(op);
        op->run_func = op_conv_2d_wino_i8xi8_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
