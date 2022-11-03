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

#include "../utils/utils.h"
#include "../utils/winograd.h"

size_t op_conv_2d_wino_i8xi8_alloc_aux_mem(op_t *op)
{
    op_conv_2d_wino_t *conv_op = (op_conv_2d_wino_t *)op;
    size_t g_num_output = conv_op->num_output / conv_op->group;
    size_t g_channel = op->input_tensors[0]->shape.dim[1] / conv_op->group;
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];

    // size_t sz = winograd_i8xi8_auxmem_size();
    size_t winograd_height
        = (ROUND(height + 2 * conv_op->pad_h - WINDOW_SIZE, WINO_STRIDE) / WINO_STRIDE + 1)
        * WINDOW_SIZE;
    size_t winograd_width
        = (ROUND(width + 2 * conv_op->pad_w - WINDOW_SIZE, WINO_STRIDE) / WINO_STRIDE + 1)
        * WINDOW_SIZE;
    size_t winograd_pad_height
        = ROUND(height + 2 * conv_op->pad_h - WINDOW_SIZE, WINO_STRIDE) + WINDOW_SIZE;
    size_t winograd_pad_width
        = ROUND(width + 2 * conv_op->pad_w - WINDOW_SIZE, WINO_STRIDE) + WINDOW_SIZE;
    size_t output_height = (height + 2 * conv_op->pad_h - KERNEL_SIZE) / CONV_STRIDE + 1;
    size_t output_width = (width + 2 * conv_op->pad_w - KERNEL_SIZE) / CONV_STRIDE + 1;

    size_t sz1 = winograd_pad_height * winograd_pad_width * g_channel * sizeof(int8_t);
    size_t sz2 = winograd_height * winograd_width * g_channel * sizeof(int8_t);
    size_t sz3 = winograd_height * winograd_width * g_num_output * sizeof(int32_t);
    size_t sz4 = winograd_height / WINDOW_SIZE * OUTPUT_SIZE * winograd_width / WINDOW_SIZE
        * OUTPUT_SIZE * g_num_output * sizeof(int32_t);

    mem_alloc(conv_op->aux_mem1, sz1);
    mem_alloc(conv_op->aux_mem2, sz2);
    mem_alloc(conv_op->aux_mem3, sz3);
    mem_alloc(conv_op->aux_mem4, sz4);
    mem_alloc(conv_op->tmp_output, sizeof(int32_t) * output_height * output_width * g_num_output);

    return sz1 + sz2 + sz3 + sz4 + sizeof(int32_t) * output_height * output_width * g_num_output;
}

void op_conv_2d_wino_i8xi8_run(op_t *op)
{
    TI(pre);
    op_conv_2d_wino_t *conv_op = (op_conv_2d_wino_t *)op;
    size_t i;
    int8_t *output_0 = mem_data(op->output_tensors[0].mem);

    float walpha = conv_op->walpha[0];
    float ialpha = conv_op->ialpha[0];
    float oalpha = conv_op->oalpha[0];

/* do it offline once */
/* alpha = result_mult / 2 ^ result_shift */
#ifdef USE_FIXED_POINT_ONLY
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif

    size_t output_cnt = shape_count(&op->output_tensors[0].shape);
    CHECK_GE(mem_sizeof(conv_op->tmp_output), output_cnt * sizeof(int32_t));
    int32_t *tmp_output = mem_cpu_data(conv_op->tmp_output);
    TO(pre, "pre");

#if defined(__aarch64__) && !defined(__APPLE__)
    // implement of winograd
    TI(winograd);

    size_t j, k;
    size_t g_channel = op->input_tensors[0]->shape.dim[1] / conv_op->group;
    size_t g_num_output = conv_op->num_output / conv_op->group;
    size_t g_kernel_count = conv_op->kernel_h * conv_op->kernel_w * g_channel * g_num_output;
    size_t i_g_chw
        = g_channel * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * g_num_output;
    const int8_t *input_0 = mem_data(op->input_tensors[0]->mem);
    const int8_t *input_1 = mem_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = NULL;

    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);

    if (3 == op->input_size) {
        input_2 = mem_data(op->input_tensors[2]->mem);
    }
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < conv_op->group; ++j) {
            winograd_k3s1_i8xi8(
                input_0, input_1, conv_op->aux_mem1, conv_op->aux_mem2, conv_op->aux_mem3,
                conv_op->aux_mem4, tmp_output, g_channel, op->input_tensors[0]->shape.dim[2],
                op->input_tensors[0]->shape.dim[3], g_num_output, conv_op->pad_h, conv_op->pad_w);
            input_0 += i_g_chw;
            input_1 += g_kernel_count;
            tmp_output += o_g_chw;
        }
        input_1 -= input1_cnt;
    }
    TO(winograd, "winograd");
#endif

    TI(bias);
#if defined(__aarch64__) && !defined(__APPLE__)
    tmp_output = mem_cpu_data(conv_op->tmp_output);
    if (op->input_size == 3) {
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < conv_op->num_output; ++j) {
                int32_t tmp = input_2[j];
                for (k = 0; k < o_hw / 8; ++k) {
                    tmp_output[0] += tmp;
                    tmp_output[1] += tmp;
                    tmp_output[2] += tmp;
                    tmp_output[3] += tmp;
                    tmp_output[4] += tmp;
                    tmp_output[5] += tmp;
                    tmp_output[6] += tmp;
                    tmp_output[7] += tmp;
                    tmp_output += 8;
                }
                for (k = 0; k < o_hw % 8; ++k) {
                    *tmp_output++ += tmp;
                }
            }
        }
    }
#endif
    TO(bias, "bias");

    TI(saturate);
    tmp_output = mem_cpu_data(conv_op->tmp_output);
    uint8_t bit = conv_op->obits[0];
    for (i = 0; i < output_cnt / 8; i++) {
        output_0[0] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[0]) * (int64_t)result_mult, result_shift), bit);
        output_0[1] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[1]) * (int64_t)result_mult, result_shift), bit);
        output_0[2] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[2]) * (int64_t)result_mult, result_shift), bit);
        output_0[3] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[3]) * (int64_t)result_mult, result_shift), bit);
        output_0[4] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[4]) * (int64_t)result_mult, result_shift), bit);
        output_0[5] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[5]) * (int64_t)result_mult, result_shift), bit);
        output_0[6] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[6]) * (int64_t)result_mult, result_shift), bit);
        output_0[7] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[7]) * (int64_t)result_mult, result_shift), bit);
        output_0 += 8;
        tmp_output += 8;
    }
    for (i = 0; i < output_cnt % 8; i++) {
        output_0[0] = ssaturate_int_by_bits(
            rshift_rn((int64_t)(tmp_output[0]) * (int64_t)result_mult, result_shift), bit);
        output_0++;
        tmp_output++;
    }
    TO(saturate, "conv_saturate");
}
