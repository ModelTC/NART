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

#include "../utils/im2col.hpp"
#include "../utils/sgemm.hpp"

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
    mem_t *temp_data; // temp mem for col in col2im
    mem_t *aux_mem_1; // temp mem for matrix A in sgemm
    mem_t *aux_mem_2; // temp mem for matrix B in sgemm
} op_deconv_2d_t;
op_deconv_2d_t *op_default_deconv_2d_tp_alloc(workspace_t *ws);
void op_default_deconv_2d_tp_config(op_t *op);
void op_default_deconv_2d_tp_destroy(op_t *op);
void op_default_deconv_2d_tp_dealloc(op_t *op);
void op_default_deconv_2d_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_deconv_2d_t *op_default_deconv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_deconv_2d_t *res = (op_deconv_2d_t *)malloc(sizeof(op_deconv_2d_t));
    memset(res, 0, sizeof(op_deconv_2d_t));
    return res;
}

void op_default_deconv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_deconv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_deconv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_deconv_2d_t *)op)->kernel_w));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_deconv_2d_t *)op)->pad_h));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_deconv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_deconv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_deconv_2d_t *)op)->stride_w));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_deconv_2d_t *)op)->group));
}

void op_default_deconv_2d_tp_destroy(op_t *op)
{
    if (((op_deconv_2d_t *)op)->aux_mem_1)
        mem_delete(((op_deconv_2d_t *)op)->aux_mem_1);
    if (((op_deconv_2d_t *)op)->aux_mem_2)
        mem_delete(((op_deconv_2d_t *)op)->aux_mem_2);
    if (((op_deconv_2d_t *)op)->temp_data)
        mem_delete(((op_deconv_2d_t *)op)->temp_data);
}

void op_default_deconv_2d_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T1, typename T2, typename T3> static void op_default_deconv_2d_blas_run(op_t *op)
{
    size_t i, j;
    const op_deconv_2d_t *deconv_op = (op_deconv_2d_t *)op;
    size_t group = deconv_op->group;
    size_t kernel_size = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group;
    size_t g_kernel_count = kernel_size * op->input_tensors[0]->shape.dim[1] / group;
    size_t i_hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t i_g_chw = (op->input_tensors[0]->shape.dim[1] / group) * i_hw;
    size_t o_g_chw = op->output_tensors[0].shape.dim[1] * op->output_tensors[0].shape.dim[2]
        * op->output_tensors[0].shape.dim[3] / group;
    const T1 *input_0 = (const T1 *)mem_cpu_data(op->input_tensors[0]->mem);
    const T2 *input_1 = (const T2 *)mem_cpu_data(op->input_tensors[1]->mem);
    T3 *output_0 = (T3 *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(T3) * shape_count(&op->output_tensors[0].shape));

    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        input_1 = (const T2 *)mem_cpu_data(op->input_tensors[1]->mem);
        for (j = 0; j < group; ++j) {
            sgemm_ATxB(
                kernel_size, i_hw, op->input_tensors[0]->shape.dim[1] / group, input_1, input_0,
                (T3 *)mem_cpu_data(deconv_op->temp_data), deconv_op->aux_mem_1,
                deconv_op->aux_mem_2);
            col2im(
                (const T3 *)mem_cpu_data(deconv_op->temp_data), output_0,
                op->output_tensors[0].shape.dim[1] / group, op->output_tensors[0].shape.dim[2],
                op->output_tensors[0].shape.dim[3], deconv_op->kernel_h, deconv_op->kernel_w,
                deconv_op->pad_h, deconv_op->pad_w, deconv_op->stride_h, deconv_op->stride_w);
            input_0 += i_g_chw;
            input_1 += g_kernel_count;
            output_0 += o_g_chw;
        }
    }
}

template <typename T1, typename T2, typename T3>
static void op_default_deconv_2d_bias_blas_run(op_t *op)
{
    size_t i, j, k;
    const op_deconv_2d_t *deconv_op = (op_deconv_2d_t *)op;
    size_t group = deconv_op->group;
    size_t kernel_size = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group;
    size_t g_kernel_count = kernel_size * op->input_tensors[0]->shape.dim[1] / group;
    size_t i_hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t i_g_chw = (op->input_tensors[0]->shape.dim[1] / group) * i_hw;
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * deconv_op->num_output / group;
    const T1 *input_0 = (const T1 *)mem_cpu_data(op->input_tensors[0]->mem);
    const T2 *input_1 = (const T2 *)mem_cpu_data(op->input_tensors[1]->mem);
    const T3 *input_2 = (const T3 *)mem_cpu_data(op->input_tensors[2]->mem);
    T3 *output_0 = (T3 *)mem_cpu_data(op->output_tensors[0].mem);

    // prepare bias data
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < deconv_op->num_output; ++j) {
            for (k = 0; k < o_hw; ++k) {
                *output_0++ = input_2[j];
            }
        }
    }
    output_0 -= op->input_tensors[0]->shape.dim[0] * deconv_op->num_output * o_hw;

    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        input_1 = (const T2 *)mem_cpu_data(op->input_tensors[1]->mem);
        for (j = 0; j < group; ++j) {
            sgemm_ATxB(
                kernel_size, i_hw, op->input_tensors[0]->shape.dim[1] / group, input_1, input_0,
                (T3 *)mem_cpu_data(deconv_op->temp_data), deconv_op->aux_mem_1,
                deconv_op->aux_mem_2);
            col2im(
                (const T3 *)mem_cpu_data(deconv_op->temp_data), output_0,
                op->output_tensors[0].shape.dim[1] / group, op->output_tensors[0].shape.dim[2],
                op->output_tensors[0].shape.dim[3], deconv_op->kernel_h, deconv_op->kernel_w,
                deconv_op->pad_h, deconv_op->pad_w, deconv_op->stride_h, deconv_op->stride_w);
            input_0 += i_g_chw;
            input_1 += g_kernel_count;
            output_0 += o_g_chw;
        }
    }
}

void op_default_deconv_2d_tp_prepare(op_t *op)
{
    int i;
    op_deconv_2d_t *deconv_op = (op_deconv_2d_t *)op;
    size_t count;
    uint32_t group = deconv_op->group;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }

    if (op->input_tensors[0]->dtype == dtFLOAT32) {
        // chaneg output dtype
        op->output_tensors[0].dtype = dtFLOAT32;
        tensor_alloc(&op->output_tensors[0]);

        if (NULL == deconv_op->aux_mem_1)
            deconv_op->aux_mem_1 = mem_new(cpu_mem_tp);
        mem_alloc(
            deconv_op->aux_mem_1,
            sizeof(float) * op->input_tensors[0]->shape.dim[1] / group * deconv_op->num_output
                * deconv_op->kernel_h * deconv_op->kernel_w / group);
        if (NULL == deconv_op->aux_mem_2)
            deconv_op->aux_mem_2 = mem_new(cpu_mem_tp);
        mem_alloc(
            deconv_op->aux_mem_2,
            sizeof(float) * op->input_tensors[0]->shape.dim[1] / group
                * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3]);

        count = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group
            * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];

        if (NULL == deconv_op->temp_data)
            deconv_op->temp_data = mem_new(cpu_mem_tp);
        mem_alloc(deconv_op->temp_data, sizeof(float) * count);
        memset(mem_cpu_data(deconv_op->temp_data), 0, sizeof(float) * count);
        if (2 == op->input_size) {
            op->run_func = op_default_deconv_2d_blas_run<float, float, float>;
        } else {
            op->run_func = op_default_deconv_2d_bias_blas_run<float, float, float>;
        }
    } else if (op->input_tensors[0]->dtype == dtUINT8 || op->input_tensors[0]->dtype == dtINT8) {
        tensor_alloc(&op->output_tensors[0]);

        if (NULL == deconv_op->aux_mem_1)
            deconv_op->aux_mem_1 = mem_new(cpu_mem_tp);
        mem_alloc(
            deconv_op->aux_mem_1,
            sizeof(uint8_t) * op->input_tensors[0]->shape.dim[1] / group * deconv_op->num_output
                * deconv_op->kernel_h * deconv_op->kernel_w / group);
        if (NULL == deconv_op->aux_mem_2)
            deconv_op->aux_mem_2 = mem_new(cpu_mem_tp);
        mem_alloc(
            deconv_op->aux_mem_2,
            sizeof(int8_t) * op->input_tensors[0]->shape.dim[1] / group
                * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3]);

        count = deconv_op->kernel_h * deconv_op->kernel_w * deconv_op->num_output / group
            * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];

        if (NULL == deconv_op->temp_data)
            deconv_op->temp_data = mem_new(cpu_mem_tp);
        mem_alloc(deconv_op->temp_data, sizeof(int32_t) * count);
        memset(mem_cpu_data(deconv_op->temp_data), 0, sizeof(int32_t) * count);

        if (op->input_tensors[0]->dtype == dtUINT8) {
            if (2 == op->input_size) {
                op->run_func = op_default_deconv_2d_blas_run<uint8_t, int8_t, int32_t>;
            } else {
                op->run_func = op_default_deconv_2d_bias_blas_run<uint8_t, int8_t, int32_t>;
            }
        } else {
            if (2 == op->input_size) {
                op->run_func = op_default_deconv_2d_blas_run<int8_t, int8_t, int32_t>;
            } else {
                op->run_func = op_default_deconv_2d_bias_blas_run<int8_t, int8_t, int32_t>;
            }
        }
    } else {
        CHECK(false);
    }
}
