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

size_t op_conv_2d_i8xi8_alloc_aux_mem(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    int M = conv_op->num_output / conv_op->group;
    int N = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    int K = conv_op->kernel_h * conv_op->kernel_w * op->input_tensors[1]->shape.dim[1];

    size_t sz = conv_op->gemm_auxmem_size(M, N, K);
    mem_alloc(conv_op->aux_mem, sz);
    mem_alloc(conv_op->temp_data, sizeof(int8_t) * N * K);
    mem_alloc(conv_op->tmp_output, sizeof(int32_t) * conv_op->num_output * N);

    return sz + sizeof(int8_t) * N * K + sizeof(int32_t) * M * N;
}

void op_conv_2d_i8xi8_run(op_t *op)
{
    TI(pre);
    size_t i, j, k;
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    size_t group = conv_op->group;
    size_t g_channel = op->input_tensors[0]->shape.dim[1] / group;
    size_t g_num_output = conv_op->num_output / group;
    size_t kernel_size = conv_op->kernel_h * conv_op->kernel_w * g_channel;
    size_t g_kernel_count = kernel_size * g_num_output;
    size_t i_g_chw
        = g_channel * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * g_num_output;
    const int8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const int8_t *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = NULL;
    if (op->input_size == 3)
        input_2 = mem_cpu_data(op->input_tensors[2]->mem);
    int8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

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

    size_t count = conv_op->kernel_h * conv_op->kernel_w
        * (op->input_tensors[0]->shape.dim[1] / group) * op->output_tensors[0].shape.dim[2]
        * op->output_tensors[0].shape.dim[3];

    CHECK_GE(mem_sizeof(conv_op->temp_data), count * sizeof(int8_t));
    // memset(mem_cpu_data(conv_op->temp_data), 0, sizeof(int8_t) * count);

    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    CHECK_GE(mem_sizeof(conv_op->tmp_output), output_cnt * sizeof(int32_t));
    int32_t *tmp_output = mem_cpu_data(conv_op->tmp_output);

#ifndef __aarch64__
    if (op->input_size == 3) {
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < conv_op->num_output; ++j) {
                for (k = 0; k < o_hw; ++k) {
                    *tmp_output++ = input_2[j];
                }
            }
        }
        tmp_output -= output_cnt;
    }
#endif

    TO(pre, "pre");
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < group; ++j) {
            TI(im2col);
            im2col_i8(
                input_0, mem_cpu_data(conv_op->temp_data), g_channel,
                op->input_tensors[0]->shape.dim[2], op->input_tensors[0]->shape.dim[3],
                conv_op->kernel_h, conv_op->kernel_w, conv_op->pad_h, conv_op->pad_w,
                conv_op->stride_h, conv_op->stride_w);
            TO(im2col, "conv_im2col");

            TI(sgemm);
            conv_op->gemm(
                g_num_output, o_hw, kernel_size, input_1, mem_cpu_data(conv_op->temp_data),
                tmp_output, conv_op->aux_mem);
            TO(sgemm, "conv_sgemm");

            input_0 += i_g_chw;
            input_1 += g_kernel_count;
            tmp_output += o_g_chw;
        }
        input_1 -= input1_cnt;
    }

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
