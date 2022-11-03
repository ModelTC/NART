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

#include "../gemm/gemm.h"
#include "../utils/im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t group;
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
        mem_t *, mem_t *);
    void (*ugemm)(
        const size_t, const size_t, const size_t, const int16_t *, const int16_t *, int32_t *,
        mem_t *, mem_t *);
    // size_t (*gemm_auxmem_size)(int, int, int);

    mem_t *temp_data; // temp mem for col in im2col
    mem_t *aux_mem_1; // temp mem for matrix A in sgemm
    mem_t *aux_mem_2; // temp mem for matrix C in sgemm
    mem_t *tmp_input0;
    mem_t *tmp_input1;
    mem_t *tmp_output;
} op_deconv_2d_group_t;
op_deconv_2d_group_t *op_quant_deconv_2d_tp_alloc(workspace_t *ws);
void op_quant_deconv_2d_tp_config(op_t *op);
void op_quant_deconv_2d_tp_destroy(op_t *op);
void op_quant_deconv_2d_tp_dealloc(op_t *op);
void op_quant_deconv_2d_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_deconv_2d_group_t *op_quant_deconv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_deconv_2d_group_t *res = (op_deconv_2d_group_t *)malloc(sizeof(op_deconv_2d_group_t));
    memset(res, 0, sizeof(op_deconv_2d_group_t));
    return res;
}

void op_quant_deconv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_deconv_2d_group_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_deconv_2d_group_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_deconv_2d_group_t *)op)->kernel_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_deconv_2d_group_t *)op)->pad_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_deconv_2d_group_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_deconv_2d_group_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_deconv_2d_group_t *)op)->stride_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_deconv_2d_group_t *)op)->group));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_deconv_2d_group_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_deconv_2d_group_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_deconv_2d_group_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_deconv_2d_group_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WZERO_POINT, dtUINT8, &len_zero_point,
        &((op_deconv_2d_group_t *)op)->wzero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WBITS, dtUINT8, &len_bits, &((op_deconv_2d_group_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_deconv_2d_group_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_deconv_2d_group_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_deconv_2d_group_t *)op)->obits));
}

void op_quant_deconv_2d_tp_destroy(op_t *op)
{
    op_deconv_2d_group_t *deconv_op = (op_deconv_2d_group_t *)op;
    if (NULL != deconv_op->aux_mem_1)
        mem_delete(deconv_op->aux_mem_1);
    if (NULL != deconv_op->aux_mem_2)
        mem_delete(deconv_op->aux_mem_2);
    if (NULL != deconv_op->temp_data) {
        mem_delete(deconv_op->temp_data);
        deconv_op->temp_data = NULL;
    }
    if (NULL != deconv_op->tmp_input0) {
        mem_delete(deconv_op->tmp_input0);
        deconv_op->tmp_input0 = NULL;
    }
    if (NULL != deconv_op->tmp_input1) {
        mem_delete(deconv_op->tmp_input1);
        deconv_op->tmp_input1 = NULL;
    }
    if (NULL != deconv_op->tmp_output) {
        mem_delete(deconv_op->tmp_output);
        deconv_op->tmp_output = NULL;
    }
}

void op_quant_deconv_2d_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_deconv_2d_run(op_t *op)
{
    size_t i, j;
    op_deconv_2d_group_t *deconv_op = (op_deconv_2d_group_t *)op;
    size_t group = deconv_op->group;
    size_t kernel_size = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group;
    size_t g_kernel_count = kernel_size * op->input_tensors[0]->shape.dim[1] / group;

    size_t i_hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t i_g_chw = (op->input_tensors[0]->shape.dim[1] / group) * i_hw;
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * deconv_op->num_output / group;

    const uint8_t *input_0 = (const uint8_t *)mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *input_1 = (const uint8_t *)mem_cpu_data(op->input_tensors[1]->mem);
    uint8_t *output_0 = (uint8_t *)mem_cpu_data(op->output_tensors[0].mem);

    float walpha = deconv_op->walpha[0];
    uint8_t wzero_point = deconv_op->wzero_point[0];

    float ialpha = deconv_op->ialpha[0];
    uint8_t izero_point = deconv_op->izero_point[0];

    float oalpha = deconv_op->oalpha[0];
    uint8_t ozero_point = deconv_op->ozero_point[0];

/* do it offline once */
/* alpha = result_mult / 2 ^ result_shift */
#ifdef USE_FIXED_POINT_ONLY
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif

    /* output = (ialpha * walpha / oalpha) * (weight - izero_point) * (input - izeo_point) */
    size_t input0_cnt = shape_count(&op->input_tensors[0]->shape);
    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    int16_t *tmp_input0 = (int16_t *)mem_cpu_data(deconv_op->tmp_input0);
    int16_t *tmp_input1 = (int16_t *)mem_cpu_data(deconv_op->tmp_input1);
    int32_t *tmp_output = (int32_t *)mem_cpu_data(deconv_op->tmp_output);
    for (i = 0; i < input0_cnt; i++) {
        tmp_input0[i] = input_0[i] - izero_point;
    }
    for (i = 0; i < input1_cnt; i++) {
        tmp_input1[i] = input_1[i] - wzero_point;
    }

    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < group; ++j) {
            deconv_op->ugemm(
                kernel_size, i_hw, op->input_tensors[0]->shape.dim[1] / group, tmp_input1,
                tmp_input0, (int32_t *)mem_cpu_data(deconv_op->temp_data), deconv_op->aux_mem_1,
                deconv_op->aux_mem_2);

            col2im_i32(
                (const int32_t *)mem_cpu_data(deconv_op->temp_data), tmp_output,
                op->output_tensors[0].shape.dim[1] / group, op->output_tensors[0].shape.dim[2],
                op->output_tensors[0].shape.dim[3], deconv_op->kernel_h, deconv_op->kernel_w,
                deconv_op->pad_h, deconv_op->pad_w, deconv_op->stride_h, deconv_op->stride_w);

            tmp_input0 += i_g_chw;
            tmp_input1 += g_kernel_count;
            tmp_output += o_g_chw;
        }
        tmp_input1 -= input1_cnt;
    }
    tmp_input0 -= input0_cnt;
    tmp_output -= output_cnt;

    uint8_t bit = deconv_op->obits[0];
    for (i = 0; i < output_cnt; i++) {
        *output_0++ = saturate_int_by_bits(
            rshift_rn((int64_t)(*(tmp_output)++) * (int64_t)result_mult, result_shift)
                + ozero_point,
            bit);
    }
    deconv_op->tmp_output -= output_cnt;
}

static void op_deconv_2d_bias_run(op_t *op)
{
    size_t i, j, k;
    op_deconv_2d_group_t *deconv_op = (op_deconv_2d_group_t *)op;
    size_t group = deconv_op->group;
    size_t kernel_size = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group;
    size_t g_kernel_count = kernel_size * op->input_tensors[0]->shape.dim[1] / group;

    size_t i_hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t i_g_chw = (op->input_tensors[0]->shape.dim[1] / group) * i_hw;
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * deconv_op->num_output / group;

    const uint8_t *input_0 = (const uint8_t *)mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *input_1 = (const uint8_t *)mem_cpu_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = (const int32_t *)mem_cpu_data(op->input_tensors[2]->mem);
    uint8_t *output_0 = (uint8_t *)mem_cpu_data(op->output_tensors[0].mem);

    float walpha = deconv_op->walpha[0];
    uint8_t wzero_point = deconv_op->wzero_point[0];

    float ialpha = deconv_op->ialpha[0];
    uint8_t izero_point = deconv_op->izero_point[0];

    float oalpha = deconv_op->oalpha[0];
    uint8_t ozero_point = deconv_op->ozero_point[0];

/* do it offline once */
/* alpha = result_mult / 2 ^ result_shift */
#ifdef USE_FIXED_POINT_ONLY
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif

    /* output = (ialpha * walpha / oalpha) * (weight - izero_point) * (input - izeo_point) */
    size_t input0_cnt = shape_count(&op->input_tensors[0]->shape);
    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    int16_t *tmp_input0 = (int16_t *)mem_cpu_data(deconv_op->tmp_input0);
    int16_t *tmp_input1 = (int16_t *)mem_cpu_data(deconv_op->tmp_input1);
    int32_t *tmp_output = (int32_t *)mem_cpu_data(deconv_op->tmp_output);
    for (i = 0; i < input0_cnt; i++) {
        tmp_input0[i] = input_0[i] - izero_point;
    }
    for (i = 0; i < input1_cnt; i++) {
        tmp_input1[i] = input_1[i] - wzero_point;
    }
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < deconv_op->num_output; ++j) {
            for (k = 0; k < o_hw; ++k) {
                *(tmp_output)++ = input_2[j];
            }
        }
    }
    tmp_output -= output_cnt;

    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < group; ++j) {
            deconv_op->ugemm(
                kernel_size, i_hw, op->input_tensors[0]->shape.dim[1] / group, tmp_input1,
                tmp_input0, (int32_t *)mem_cpu_data(deconv_op->temp_data), deconv_op->aux_mem_1,
                deconv_op->aux_mem_2);

            col2im_i32(
                (const int32_t *)mem_cpu_data(deconv_op->temp_data), tmp_output,
                op->output_tensors[0].shape.dim[1] / group, op->output_tensors[0].shape.dim[2],
                op->output_tensors[0].shape.dim[3], deconv_op->kernel_h, deconv_op->kernel_w,
                deconv_op->pad_h, deconv_op->pad_w, deconv_op->stride_h, deconv_op->stride_w);

            tmp_input0 += i_g_chw;
            tmp_input1 += g_kernel_count;
            tmp_output += o_g_chw;
        }
        tmp_input1 -= input1_cnt;
    }
    tmp_input0 -= input0_cnt;
    tmp_output -= output_cnt;

    uint8_t bit = deconv_op->obits[0];
    for (i = 0; i < output_cnt; i++) {
        *output_0++ = saturate_int_by_bits(
            rshift_rn((int64_t)(*(tmp_output)++) * (int64_t)result_mult, result_shift)
                + ozero_point,
            bit);
    }
    tmp_output -= output_cnt;
}

void op_quant_deconv_2d_tp_prepare(op_t *op)
{
    int i;
    op_deconv_2d_group_t *deconv_op = (op_deconv_2d_group_t *)op;
    uint32_t group = deconv_op->group;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    /* alloc temp memory */
    if (NULL == deconv_op->aux_mem_1)
        deconv_op->aux_mem_1 = mem_new(cpu_mem_tp);
    mem_alloc(
        deconv_op->aux_mem_1,
        sizeof(int16_t) * op->input_tensors[0]->shape.dim[1] / group * deconv_op->num_output
            * deconv_op->kernel_h * deconv_op->kernel_w / group);
    if (NULL == deconv_op->aux_mem_2)
        deconv_op->aux_mem_2 = mem_new(cpu_mem_tp);
    mem_alloc(
        deconv_op->aux_mem_2,
        sizeof(int16_t) * shape_count_per_batch(&op->input_tensors[0]->shape) / group);

    size_t count = deconv_op->kernel_h * deconv_op->kernel_w
        * (op->input_tensors[0]->shape.dim[1] / group) * op->output_tensors[0].shape.dim[2]
        * op->output_tensors[0].shape.dim[3];

    if (NULL == deconv_op->temp_data)
        deconv_op->temp_data = mem_new(cpu_mem_tp);
    mem_alloc(deconv_op->temp_data, sizeof(int32_t) * count);
    if (NULL == deconv_op->tmp_input0)
        deconv_op->tmp_input0 = mem_new(cpu_mem_tp);
    mem_alloc(deconv_op->tmp_input0, sizeof(int16_t) * shape_count(&op->input_tensors[0]->shape));
    if (NULL == deconv_op->tmp_input1)
        deconv_op->tmp_input1 = mem_new(cpu_mem_tp);
    mem_alloc(deconv_op->tmp_input1, sizeof(int16_t) * shape_count(&op->input_tensors[1]->shape));
    if (NULL == deconv_op->tmp_output)
        deconv_op->tmp_output = mem_new(cpu_mem_tp);
    mem_alloc(deconv_op->tmp_output, sizeof(int32_t) * shape_count(&op->output_tensors[0].shape));

    // deconv_op->temp_data = (int32_t *)malloc(sizeof(int32_t) * count);
    // deconv_op->tmp_input1 = (int16_t *)malloc(sizeof(int16_t) *
    // shape_count(&op->input_tensors[1]->shape)); deconv_op->tmp_output = (int32_t
    // *)malloc(sizeof(int32_t) * shape_count(&op->output_tensors[0].shape));

    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        if (op->input_size == 2) {
            op->run_func = op_deconv_2d_run;
        } else {
            op->run_func = op_deconv_2d_bias_run;
        }
        deconv_op->ugemm = gemm_ATxB_i16xi16;
        break;
    default:
        CHECK(false);
        break;
    }
}
