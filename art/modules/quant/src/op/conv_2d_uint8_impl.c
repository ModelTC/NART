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

size_t op_conv_2d_u8xu8_alloc_aux_mem(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    int M = conv_op->num_output / conv_op->group;
    int N = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    int K = conv_op->kernel_h * conv_op->kernel_w * op->input_tensors[1]->shape.dim[1];

    size_t sz = conv_op->gemm_auxmem_size(M, N, K);
    mem_alloc(conv_op->aux_mem, sz);
    mem_alloc(
        conv_op->temp_data,
        sizeof(int16_t) * N * K + sizeof(int16_t) * shape_count(&op->input_tensors[0]->shape)
            + sizeof(int16_t) * shape_count(&op->input_tensors[1]->shape));
    mem_alloc(conv_op->tmp_output, sizeof(int32_t) * conv_op->num_output * N);

    return sz + sizeof(int16_t) * N * K
        + sizeof(int16_t) * shape_count(&op->input_tensors[0]->shape)
        + sizeof(int16_t) * shape_count(&op->input_tensors[0]->shape) + sizeof(int32_t) * M * N;
}

void op_conv_2d_u8xu8_run(op_t *op)
{
    size_t i, j, k;
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    // size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t group = conv_op->group;
    size_t g_channel = op->input_tensors[0]->shape.dim[1] / group;
    size_t g_num_output = conv_op->num_output / group;
    size_t kernel_size = conv_op->kernel_h * conv_op->kernel_w * g_channel;
    size_t g_kernel_count = kernel_size * g_num_output;
    size_t i_g_chw
        = g_channel * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_g_chw = o_hw * g_num_output;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const int32_t *input_2 = NULL;
    if (3 == op->input_size) {
        input_2 = mem_cpu_data(op->input_tensors[2]->mem);
    }
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    float walpha = conv_op->walpha[0];
    uint8_t wzero_point = conv_op->wzero_point[0];

    float ialpha = conv_op->ialpha[0];
    uint8_t izero_point = conv_op->izero_point[0];

    float oalpha = conv_op->oalpha[0];
    uint8_t ozero_point = conv_op->ozero_point[0];

/* do it offline once */
/* alpha = result_mult / 2 ^ result_shift */
#ifdef USE_FIXED_POINT_ONLY
    uint32_t result_mult = 0;
    int8_t result_shift = 0;
    get_quant_scale_int32(ialpha * walpha / oalpha, &result_mult, &result_shift);
#endif

    /*
    printf("i: %f %u\n", ialpha, izero_point);
    printf("w: %f %u\n", walpha, wzero_point);
    printf("o: %f %u\n", oalpha, ozero_point);
    printf("p: %f %u %i\n", ialpha * walpha / oalpha, result_mult, result_shift);
    */

    // float* output_0_temp = output_0;

    // temp test
    size_t count = conv_op->kernel_h * conv_op->kernel_w
        * (op->input_tensors[0]->shape.dim[1] / group) * op->output_tensors[0].shape.dim[2]
        * op->output_tensors[0].shape.dim[3];

    size_t input0_cnt = shape_count(&op->input_tensors[0]->shape);
    size_t input1_cnt = shape_count(&op->input_tensors[1]->shape);
    size_t output_cnt = shape_count(&op->output_tensors[0].shape);

    CHECK_GE(
        mem_sizeof(conv_op->temp_data),
        count * sizeof(int16_t) + input0_cnt * sizeof(int16_t) + input1_cnt * sizeof(int16_t));
    int16_t *temp_data = mem_cpu_data(conv_op->temp_data);
    int16_t *tmp_input0 = temp_data + count;
    int16_t *tmp_input1 = tmp_input0 + input0_cnt;
    // memset(mem_cpu_data(conv_op->temp_data), 0, sizeof(int16_t) * count);

    CHECK_GE(mem_sizeof(conv_op->tmp_output), output_cnt * sizeof(int32_t));
    int32_t *tmp_output = mem_cpu_data(conv_op->tmp_output);

    for (i = 0; i < input0_cnt; i++) {
        tmp_input0[i] = input_0[i] - izero_point;
    }
    for (i = 0; i < input1_cnt; i++) {
        tmp_input1[i] = input_1[i] - wzero_point;
    }

#ifndef __aarch64__
    if (op->input_size == 3) {
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < conv_op->num_output; ++j) {
                for (k = 0; k < o_hw; ++k) {
                    *tmp_output++ = input_2[j];
                }
            }
        }
    }
#endif

    tmp_output = mem_cpu_data(conv_op->tmp_output);
    for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
        for (j = 0; j < group; ++j) {
            im2col_i16(
                tmp_input0, temp_data, g_channel, op->input_tensors[0]->shape.dim[2],
                op->input_tensors[0]->shape.dim[3], conv_op->kernel_h, conv_op->kernel_w,
                conv_op->pad_h, conv_op->pad_w, conv_op->stride_h, conv_op->stride_w);

            conv_op->ugemm(
                g_num_output, o_hw, kernel_size, tmp_input1, temp_data, tmp_output,
                conv_op->aux_mem);

            tmp_input0 += i_g_chw;
            tmp_input1 += g_kernel_count;
            tmp_output += o_g_chw;
        }
        tmp_input1 -= input1_cnt;
    }

#if defined(__aarch64__) && !defined(__APPLE__)
    if (op->input_size == 3) {
        tmp_output = mem_cpu_data(conv_op->tmp_output);
        for (i = 0; i < op->input_tensors[0]->shape.dim[0]; ++i) {
            for (j = 0; j < conv_op->num_output; ++j) {
                for (k = 0; k < o_hw; ++k) {
                    *tmp_output++ += input_2[j];
                }
            }
        }
    }
#endif
    tmp_output = mem_cpu_data(conv_op->tmp_output);

    for (i = 0; i < output_cnt; i++) {
        *output_0++ = saturate_int_by_bits(
            rshift_rn((int64_t)(*(tmp_output)++) * (int64_t)result_mult, result_shift)
                + ozero_point,
            conv_op->obits[0]);
    }
}
