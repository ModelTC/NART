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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "winograd.h"

#if defined(__aarch64__) && !defined(__APPLE__)
void winograd_k3s1_i8xi8(
    const int8_t *input, const int8_t *weights, mem_t *aux_mem1, mem_t *aux_mem2, mem_t *aux_mem3,
    mem_t *aux_mem4, int32_t *output, size_t channel, size_t height, size_t width,
    size_t num_output, size_t pad_h, size_t pad_w)
{

    size_t i, j;
    size_t winograd_height
        = (ROUND(height + 2 * pad_h - WINDOW_SIZE, WINO_STRIDE) / WINO_STRIDE + 1) * WINDOW_SIZE;
    size_t winograd_width
        = (ROUND(width + 2 * pad_w - WINDOW_SIZE, WINO_STRIDE) / WINO_STRIDE + 1) * WINDOW_SIZE;
    size_t winograd_pad_height = ROUND(height + 2 * pad_h - WINDOW_SIZE, WINO_STRIDE) + WINDOW_SIZE;
    size_t winograd_pad_width = ROUND(width + 2 * pad_w - WINDOW_SIZE, WINO_STRIDE) + WINDOW_SIZE;
    size_t output_height = (height + 2 * pad_h - KERNEL_SIZE) / CONV_STRIDE + 1;
    size_t output_width = (width + 2 * pad_w - KERNEL_SIZE) / CONV_STRIDE + 1;

    int8_t *winograd_pad_input = mem_cpu_data(aux_mem1);
    int8_t *winograd_input = mem_cpu_data(aux_mem2);
    int32_t *winograd_output = mem_cpu_data(aux_mem3);
    int32_t *winograd_pad_output = mem_cpu_data(aux_mem4);

    // prepare pad input
    int8_t *temp = winograd_pad_input + winograd_pad_width * pad_h;
    for (i = 0; i < channel; ++i) {
        for (j = 0; j < height; ++j) {
            // memcpy(temp + pad_w, input, sizeof(int8_t) * width);
            memcpy_neon64(temp + pad_w, input, sizeof(int8_t) * width);
            temp += winograd_pad_width;
            input += width;
        }
        temp += (winograd_pad_height - height) * winograd_pad_width;
    }

    winograd_input_process(
        winograd_pad_input, winograd_input, winograd_pad_height, winograd_pad_width, channel);
    eltwise_mult_pseudo_gemm(
        winograd_input, weights, winograd_output, winograd_height, winograd_width, channel,
        num_output);
    winograd_output_process(
        winograd_output, winograd_pad_output, output, output_height, output_width, winograd_height,
        winograd_width, num_output);

    return;
}

void winograd_input_process(
    int8_t *winograd_pad_input, int8_t *winograd_input, size_t winograd_pad_height,
    size_t winograd_pad_width, size_t channel)
{
    size_t i, j, k;
    for (i = 0; i < winograd_pad_height - KERNEL_SIZE; i += WINO_STRIDE) {
        for (j = 0; j < winograd_pad_width - KERNEL_SIZE; j += WINO_STRIDE) {
            for (k = 0; k < channel; ++k) {
                const int8_t block0 = winograd_pad_input[0];
                const int8_t block1 = winograd_pad_input[1];
                const int8_t block2 = winograd_pad_input[2];
                const int8_t block3 = winograd_pad_input[3];
                const int8_t block4 = winograd_pad_input[0 + winograd_pad_width];
                const int8_t block5 = winograd_pad_input[1 + winograd_pad_width];
                const int8_t block6 = winograd_pad_input[2 + winograd_pad_width];
                const int8_t block7 = winograd_pad_input[3 + winograd_pad_width];
                const int8_t block8
                    = winograd_pad_input[0 + winograd_pad_width + winograd_pad_width];
                const int8_t block9
                    = winograd_pad_input[1 + winograd_pad_width + winograd_pad_width];
                const int8_t block10
                    = winograd_pad_input[2 + winograd_pad_width + winograd_pad_width];
                const int8_t block11
                    = winograd_pad_input[3 + winograd_pad_width + winograd_pad_width];
                const int8_t block12 = winograd_pad_input
                    [0 + winograd_pad_width + winograd_pad_width + winograd_pad_width];
                const int8_t block13 = winograd_pad_input
                    [1 + winograd_pad_width + winograd_pad_width + winograd_pad_width];
                const int8_t block14 = winograd_pad_input
                    [2 + winograd_pad_width + winograd_pad_width + winograd_pad_width];
                const int8_t block15 = winograd_pad_input
                    [3 + winograd_pad_width + winograd_pad_width + winograd_pad_width];
                // row 1
                winograd_input[0] = block0 - block8 - block2 + block10;
                winograd_input[1] = block1 - block9 + block2 - block10;
                winograd_input[2] = block2 - block10 - block1 + block9;
                winograd_input[3] = block1 - block9 - block3 + block11;
                // row 2
                winograd_input[4] = block4 + block8 - block6 - block10;
                winograd_input[5] = block5 + block9 + block6 + block10;
                winograd_input[6] = block6 + block10 - block5 - block9;
                winograd_input[7] = block5 + block9 - block7 - block11;
                // row 3
                winograd_input[8] = -block4 + block8 + block6 - block10;
                winograd_input[9] = -block5 + block9 - block6 + block10;
                winograd_input[10] = -block6 + block10 + block5 - block9;
                winograd_input[11] = -block5 + block9 + block7 - block11;
                // row 4
                winograd_input[12] = block4 - block12 - block6 + block14;
                winograd_input[13] = block5 - block13 + block6 - block14;
                winograd_input[14] = block6 - block14 - block5 + block13;
                winograd_input[15] = block5 - block13 - block7 + block15;
                // change to next channel
                winograd_pad_input += winograd_pad_height * winograd_pad_width;
                winograd_input += WINDOW_SIZE * WINDOW_SIZE;
            }
            winograd_pad_input -= winograd_pad_height * winograd_pad_width * channel - WINO_STRIDE;
        }
        winograd_pad_input += (WINO_STRIDE - 1) * winograd_pad_width + (WINDOW_SIZE - WINO_STRIDE);
    }

    return;
}

void eltwise_mult_pseudo_gemm(
    int8_t *winograd_input, const int8_t *winograd_weights, int32_t *winograd_output,
    size_t winograd_height, size_t winograd_width, size_t channel, size_t num_output)
{
    size_t block_num = winograd_height / WINDOW_SIZE * winograd_width / WINDOW_SIZE;

    // clang-format off
__asm__ __volatile__(
"int8_num_output4_loop_v4_%=:\n\t"
"cmp %[num_output], #4\n\t"
"blt int8_finish_num_output4_loop_v4_%=\n\t"

"mov x0, %[block_num]\n\t"
"mov x1, %[weights_addr1]\n\t"
"mov x2, %[weights_addr2]\n\t"
"mov x3, %[weights_addr3]\n\t"
"mov x4, %[weights_addr4]\n\t"

"int8_num_output4_block_num1_loop_v4_%=:\n\t"
"cmp x0, #0\n\t"
"beq int8_finish_num_output4_block_num1_loop_v4_%=\n\t"

"mov x5, %[channel]\n\t"
"movi v16.2d, #0\n\t"
"movi v17.2d, #0\n\t"
"movi v18.2d, #0\n\t"
"movi v19.2d, #0\n\t"
"movi v20.2d, #0\n\t"
"movi v21.2d, #0\n\t"
"movi v22.2d, #0\n\t"
"movi v23.2d, #0\n\t"
"movi v24.2d, #0\n\t"
"movi v25.2d, #0\n\t"
"movi v26.2d, #0\n\t"
"movi v27.2d, #0\n\t"
"movi v28.2d, #0\n\t"
"movi v29.2d, #0\n\t"
"movi v30.2d, #0\n\t"
"movi v31.2d, #0\n\t"

#ifdef WEIGHTS_INT5_5
"int8_num_output4_block_num1_channel8_loop_v4_%=:\n\t"
"cmp x5, #8\n\t"
"blt int8_finish_num_output4_block_num1_channel8_loop_v4_%=\n\t"

// channel 0
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smull v8.8h, v0.8b, v4.8b\n\t"
"smull2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smull v10.8h, v1.8b, v4.8b\n\t"
"smull2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smull v12.8h, v2.8b, v4.8b\n\t"
"smull2 v13.8h, v2.16b, v4.16b\n\t"
"smull v14.8h, v3.8b, v4.8b\n\t"
"smull2 v15.8h, v3.16b, v4.16b\n\t"
// channel 1
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 2
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 3
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 4
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 5
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 6
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 7
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"saddw v16.4s, v16.4s, v8.4h\n\t"
"saddw2 v17.4s, v17.4s, v8.8h\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"saddw v18.4s, v18.4s, v9.4h\n\t"
"saddw2 v19.4s, v19.4s, v9.8h\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"saddw v20.4s, v20.4s, v10.4h\n\t"
"saddw2 v21.4s, v21.4s, v10.8h\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"saddw v22.4s, v22.4s, v11.4h\n\t"
"saddw2 v23.4s, v23.4s, v11.8h\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
"saddw v24.4s, v24.4s, v12.4h\n\t"
"saddw2 v25.4s, v25.4s, v12.8h\n\t"
"saddw v26.4s, v26.4s, v13.4h\n\t"
"saddw2 v27.4s, v27.4s, v13.8h\n\t"
"saddw v28.4s, v28.4s, v14.4h\n\t"
"saddw2 v29.4s, v29.4s, v14.8h\n\t"
"saddw v30.4s, v30.4s, v15.4h\n\t"
"saddw2 v31.4s, v31.4s, v15.8h\n\t"

"sub x5, x5, #8\n\t"
"b int8_num_output4_block_num1_channel8_loop_v4_%=\n\t"
"int8_finish_num_output4_block_num1_channel8_loop_v4_%=:\n\t"
#endif

"int8_num_output4_block_num1_channel4_loop_v4_%=:\n\t"
"cmp x5, #4\n\t"
"blt int8_finish_num_output4_block_num1_channel4_loop_v4_%=\n\t"

// channel 0
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smull v8.8h, v0.8b, v4.8b\n\t"
"smull2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smull v10.8h, v1.8b, v4.8b\n\t"
"smull2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smull v12.8h, v2.8b, v4.8b\n\t"
"smull2 v13.8h, v2.16b, v4.16b\n\t"
"smull v14.8h, v3.8b, v4.8b\n\t"
"smull2 v15.8h, v3.16b, v4.16b\n\t"
// channel 1
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 2
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
// channel 3
"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal v8.8h, v0.8b, v4.8b\n\t"
"smlal2 v9.8h, v0.16b, v4.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal v10.8h, v1.8b, v4.8b\n\t"
"smlal2 v11.8h, v1.16b, v4.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"saddw v16.4s, v16.4s, v8.4h\n\t"
"saddw2 v17.4s, v17.4s, v8.8h\n\t"
"smlal v12.8h, v2.8b, v4.8b\n\t"
"saddw v18.4s, v18.4s, v9.4h\n\t"
"saddw2 v19.4s, v19.4s, v9.8h\n\t"
"smlal2 v13.8h, v2.16b, v4.16b\n\t"
"saddw v20.4s, v20.4s, v10.4h\n\t"
"saddw2 v21.4s, v21.4s, v10.8h\n\t"
"smlal v14.8h, v3.8b, v4.8b\n\t"
"saddw v22.4s, v22.4s, v11.4h\n\t"
"saddw2 v23.4s, v23.4s, v11.8h\n\t"
"smlal2 v15.8h, v3.16b, v4.16b\n\t"
"saddw v24.4s, v24.4s, v12.4h\n\t"
"saddw2 v25.4s, v25.4s, v12.8h\n\t"
"saddw v26.4s, v26.4s, v13.4h\n\t"
"saddw2 v27.4s, v27.4s, v13.8h\n\t"
"saddw v28.4s, v28.4s, v14.4h\n\t"
"saddw2 v29.4s, v29.4s, v14.8h\n\t"
"saddw v30.4s, v30.4s, v15.4h\n\t"
"saddw2 v31.4s, v31.4s, v15.8h\n\t"

"sub x5, x5, #4\n\t"
"b int8_num_output4_block_num1_channel4_loop_v4_%=\n\t"
"int8_finish_num_output4_block_num1_channel4_loop_v4_%=:\n\t"

"movi v8.2d, #0\n\t"
"movi v9.2d, #0\n\t"
"movi v10.2d, #0\n\t"
"movi v11.2d, #0\n\t"
"movi v12.2d, #0\n\t"
"movi v13.2d, #0\n\t"
"movi v14.2d, #0\n\t"
"movi v15.2d, #0\n\t"

"int8_num_output4_block_num1_channel1_loop_v4_%=:\n\t"
"cmp x5, #0\n\t"
"beq int8_finish_num_output4_block_num1_channel1_loop_v4_%=\n\t"

"ld1 {v4.16b}, [%[input_addr]], #16\n\t"
"ld1 {v0.16b}, [x1], %[weights_offset]\n\t"
"ld1 {v1.16b}, [x2], %[weights_offset]\n\t"
"smlal  v8.8h, v4.8b, v0.8b\n\t"
"smlal2 v9.8h, v4.16b, v0.16b\n\t"
"ld1 {v2.16b}, [x3], %[weights_offset]\n\t"
"smlal  v10.8h, v4.8b, v1.8b\n\t"
"smlal2 v11.8h, v4.16b, v1.16b\n\t"
"ld1 {v3.16b}, [x4], %[weights_offset]\n\t"
"smlal  v12.8h, v4.8b, v2.8b\n\t"
"smlal2 v13.8h, v4.16b, v2.16b\n\t"
"smlal  v14.8h, v4.8b, v3.8b\n\t"
"smlal2 v15.8h, v4.16b, v3.16b\n\t"

"sub x5, x5, #1\n\t"
"b int8_num_output4_block_num1_channel1_loop_v4_%=\n\t"
"int8_finish_num_output4_block_num1_channel1_loop_v4_%=:\n\t"

"saddw v16.4s, v16.4s, v8.4h\n\t"
"saddw2 v17.4s, v17.4s, v8.8h\n\t"
"saddw v18.4s, v18.4s, v9.4h\n\t"
"saddw2 v19.4s, v19.4s, v9.8h\n\t"
"saddw v20.4s, v20.4s, v10.4h\n\t"
"saddw2 v21.4s, v21.4s, v10.8h\n\t"
"st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output_addr1]], #64\n\t"
"saddw v22.4s, v22.4s, v11.4h\n\t"
"saddw2 v23.4s, v23.4s, v11.8h\n\t"
"saddw v24.4s, v24.4s, v12.4h\n\t"
"saddw2 v25.4s, v25.4s, v12.8h\n\t"
"st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[output_addr2]], #64\n\t"
"saddw v26.4s, v26.4s, v13.4h\n\t"
"saddw2 v27.4s, v27.4s, v13.8h\n\t"
"saddw v28.4s, v28.4s, v14.4h\n\t"
"saddw2 v29.4s, v29.4s, v14.8h\n\t"
"st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output_addr3]], #64\n\t"
"saddw v30.4s, v30.4s, v15.4h\n\t"
"saddw2 v31.4s, v31.4s, v15.8h\n\t"
"mov x1, %[weights_addr1]\n\t"
"mov x2, %[weights_addr2]\n\t"
"st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output_addr4]], #64\n\t"
"mov x3, %[weights_addr3]\n\t"
"mov x4, %[weights_addr4]\n\t"
"sub x0, x0, #1\n\t"
"b int8_num_output4_block_num1_loop_v4_%=\n\t"
"int8_finish_num_output4_block_num1_loop_v4_%=:\n\t"

"add %[weights_addr1], %[weights_addr1], %[weights_4num_output_offset]\n\t"
"add %[weights_addr2], %[weights_addr2], %[weights_4num_output_offset]\n\t"
"add %[weights_addr3], %[weights_addr3], %[weights_4num_output_offset]\n\t"
"add %[weights_addr4], %[weights_addr4], %[weights_4num_output_offset]\n\t"
"add %[output_addr1], %[output_addr1], %[output_tri_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_tri_offset]\n\t"
"add %[output_addr3], %[output_addr3], %[output_tri_offset]\n\t"
"add %[output_addr4], %[output_addr4], %[output_tri_offset]\n\t"
"sub %[input_addr], %[input_addr], %[input_resume_offset]\n\t"
"sub %[num_output], %[num_output], 4\n\t"
"b int8_num_output4_loop_v4_%=\n\t"
"int8_finish_num_output4_loop_v4_%=:\n\t"
: //output
: //input
[channel] "r" (channel), [num_output] "r" (num_output), [block_num] "r" (block_num),
[weights_addr1] "r" (winograd_weights),
[weights_addr2] "r" (winograd_weights + WINDOW_SIZE * WINDOW_SIZE * 1),
[weights_addr3] "r" (winograd_weights + WINDOW_SIZE * WINDOW_SIZE * 2),
[weights_addr4] "r" (winograd_weights + WINDOW_SIZE * WINDOW_SIZE * 3),
[input_addr] "r" (winograd_input),
[output_addr1] "r" (winograd_output),
[output_addr2] "r" (winograd_output + winograd_height * winograd_width * 1),
[output_addr3] "r" (winograd_output + winograd_height * winograd_width * 2),
[output_addr4] "r" (winograd_output + winograd_height * winograd_width * 3),
[weights_offset] "r" (WINDOW_SIZE * WINDOW_SIZE * 4 * sizeof(int8_t)),
[weights_4num_output_offset] "r" (WINDOW_SIZE * WINDOW_SIZE * channel * 4 * sizeof(int8_t)),
[input_resume_offset] "r" (winograd_height * winograd_width * channel * sizeof(int8_t)),
// [output_offset] "r" (winograd_height * winograd_width * sizeof(int32_t)),
[output_tri_offset] "r" (winograd_height * winograd_width * sizeof(int32_t) * 3)
: //clobbers
"cc", "memory", "x0", "x1", "x2", "x3", "x4", "x5",
"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
"v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
"v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
"v25", "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on

    return;
}

void winograd_output_process(
    int32_t *winograd_output, int32_t *winograd_pad_output, int32_t *output, size_t output_height,
    size_t output_width, size_t winograd_height, size_t winograd_width, size_t num_output)
{
    size_t block_num = winograd_height / WINDOW_SIZE * winograd_width / WINDOW_SIZE;

    // clang-format off
__asm__ __volatile__(
"mov x0, %[input_addr]\n\t"
"mov x1, %[output_addr]\n\t"
"mov x2, %[block_num_w]\n\t"
"int8_output_proc_dim4_loop_v6_%=:\n\t"
"cmp %[dim], #4\n\t"
"blt int8_finish_output_proc_dim4_loop_v6_%=\n\t"

"ld1 {v14.4s}, [x0], #16\n\t"
"ld1 {v15.4s}, [%[input_addr2]], #16\n\t"
"ld1 {v16.4s}, [%[input_addr3]], #16\n\t"
"ld1 {v17.4s}, [%[input_addr4]], #16\n\t"

"trn1 v28.4s, v14.4s, v15.4s\n\t"
"ld1 {v18.4s}, [x0], #16\n\t"
"trn2 v29.4s, v14.4s, v15.4s\n\t"
"ld1 {v19.4s}, [%[input_addr2]], #16\n\t"

"trn1 v30.4s, v16.4s, v17.4s\n\t"
"trn2 v31.4s, v16.4s, v17.4s\n\t"

"ld1 {v20.4s}, [%[input_addr3]], #16\n\t"
"trn1 v0.2d, v28.2d, v30.2d\n\t"
"trn2 v2.2d, v28.2d, v30.2d\n\t"

"ld1 {v21.4s}, [%[input_addr4]], #16\n\t"
"trn1 v1.2d, v29.2d, v31.2d\n\t"
"trn2 v3.2d, v29.2d, v31.2d\n\t"
"ld1 {v22.4s}, [x0], #16\n\t"

"add v16.4s, v0.4s, v1.4s\n\t"
"trn1 v28.4s, v18.4s, v19.4s\n\t"
"sub v17.4s, v1.4s, v2.4s\n\t"
"trn2 v29.4s, v18.4s, v19.4s\n\t"
"ld1 {v23.4s}, [%[input_addr2]], #16\n\t"

"add v16.4s, v16.4s, v2.4s\n\t"
"trn1 v30.4s, v20.4s, v21.4s\n\t"
"sub v17.4s, v17.4s, v3.4s\n\t"
"trn2 v31.4s, v20.4s, v21.4s\n\t"
"ld1 {v24.4s}, [%[input_addr3]], #16\n\t"

"trn1 v4.2d, v28.2d, v30.2d\n\t"
"trn2 v6.2d, v28.2d, v30.2d\n\t"
"ld1 {v25.4s}, [%[input_addr4]], #16\n\t"

"trn1 v5.2d, v29.2d, v31.2d\n\t"
"add v16.4s, v16.4s, v4.4s\n\t"
"trn2 v7.2d, v29.2d, v31.2d\n\t"

"sub v17.4s, v17.4s, v6.4s\n\t"
"ld1 {v26.4s}, [x0], #16\n\t"

"add v18.4s, v4.4s, v5.4s\n\t"
"trn1 v20.4s, v22.4s, v23.4s\n\t"
"sub v19.4s, v5.4s, v6.4s\n\t"
"trn2 v21.4s, v22.4s, v23.4s\n\t"

"add v16.4s, v16.4s, v5.4s\n\t"
"ld1 {v27.4s}, [%[input_addr2]], #16\n\t"

"add v17.4s, v17.4s, v5.4s\n\t"
"trn1 v22.4s, v24.4s, v25.4s\n\t"
"add v18.4s, v18.4s, v6.4s\n\t"
"trn2 v23.4s, v24.4s, v25.4s\n\t"
"sub v19.4s, v19.4s, v7.4s\n\t"

"ld1 {v28.4s}, [%[input_addr3]], #16\n\t"

"add v16.4s, v16.4s, v6.4s\n\t"
"trn1 v8.2d, v20.2d, v22.2d\n\t"
"sub v17.4s, v17.4s, v7.4s\n\t"
"trn2 v10.2d, v20.2d, v22.2d\n\t"

"ld1 {v29.4s}, [%[input_addr4]], #16\n\t"

"sub v18.4s, v18.4s, v8.4s\n\t"
"trn1 v9.2d, v21.2d, v23.2d\n\t"
"add v16.4s, v16.4s, v8.4s\n\t"
"trn2 v11.2d, v21.2d, v23.2d\n\t"

"trn1 v24.4s, v26.4s, v27.4s\n\t"
"add v17.4s, v17.4s, v9.4s\n\t"
"trn2 v25.4s, v26.4s, v27.4s\n\t"
"sub v19.4s, v19.4s, v9.4s\n\t"
"trn1 v26.4s, v28.4s, v29.4s\n\t"
"sub v18.4s, v18.4s, v9.4s\n\t"
"trn2 v27.4s, v28.4s, v29.4s\n\t"

"add v19.4s, v19.4s, v10.4s\n\t"
"trn1 v12.2d, v24.2d, v26.2d\n\t"
"add v16.4s, v16.4s, v9.4s\n\t"
"trn2 v14.2d, v24.2d, v26.2d\n\t"
"sub v18.4s, v18.4s, v10.4s\n\t"
"trn1 v13.2d, v25.2d, v27.2d\n\t"
"add v19.4s, v19.4s, v11.4s\n\t"
"trn2 v15.2d, v25.2d, v27.2d\n\t"

"sub v18.4s, v18.4s, v12.4s\n\t"
"sub v19.4s, v19.4s, v13.4s\n\t"

"sub v17.4s, v17.4s, v10.4s\n\t"
"sub v18.4s, v18.4s, v13.4s\n\t"
"add v19.4s, v19.4s, v14.4s\n\t"

"add v16.4s, v16.4s, v10.4s\n\t"
"sub v17.4s, v17.4s, v11.4s\n\t"
"sub v18.4s, v18.4s, v14.4s\n\t"
"add v19.4s, v19.4s, v15.4s\n\t"

"trn1 v0.4s, v16.4s, v17.4s\n\t"
"trn2 v1.4s, v16.4s, v17.4s\n\t"
"trn1 v2.4s, v18.4s, v19.4s\n\t"
"trn2 v3.4s, v18.4s, v19.4s\n\t"

"trn1 v4.2d, v0.2d, v1.2d\n\t"
"trn2 v5.2d, v0.2d, v1.2d\n\t"
"trn1 v6.2d, v2.2d, v3.2d\n\t"
"trn2 v7.2d, v2.2d, v3.2d\n\t"

"sub %[dim], %[dim], #4\n\t"
"add x0, x0, %[input_tri_offset]\n\t"
"add %[input_addr2], %[input_addr2], %[input_tri_offset]\n\t"
"add %[input_addr3], %[input_addr3], %[input_tri_offset]\n\t"
"add %[input_addr4], %[input_addr4], %[input_tri_offset]\n\t"
// "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64\n\t"
// "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64\n\t"
// "b int8_output_proc_dim4_loop_v6_%=\n\t"

"int8_output_proc_dim_branch_v6_%=:\n\t"
"cmp x2, #4\n\t"
"bge int8_output_proc_dim4_branch_v6_%=\n\t"
"cmp x2, #3\n\t"
"bge int8_output_proc_dim3_branch_v6_%=\n\t"
"cmp x2, #2\n\t"
"bge int8_output_proc_dim2_branch_v6_%=\n\t"
"cmp x2, #1\n\t"
"bge int8_output_proc_dim1_branch_v6_%=\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"mov x2, %[block_num_w]\n\t"
"b int8_output_proc_dim_branch_v6_%=\n\t"

"int8_output_proc_dim4_branch_v6_%=:\n\t"
"st1 {v4.4s, v5.4s}, [x1], #32\n\t"
"st1 {v6.4s, v7.4s}, [%[output_addr2]], #32\n\t"
"sub x2, x2, #4\n\t"
"b int8_output_proc_dim4_loop_v6_%=\n\t"

"int8_output_proc_dim3_branch_v6_%=:\n\t"
"st1 {v4.4s}, [x1], #16\n\t"
"st1 {v6.4s}, [%[output_addr2]], #16\n\t"
"st1 {v5.d}[0], [x1], #8\n\t"
"st1 {v7.d}[0], [%[output_addr2]], #8\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"st1 {v5.d}[1], [x1], #8\n\t"
"st1 {v7.d}[1], [%[output_addr2]], #8\n\t"
"sub x2, %[block_num_w], #1\n\t"
"b int8_output_proc_dim4_loop_v6_%=\n\t"

"int8_output_proc_dim2_branch_v6_%=:\n\t"
"st1 {v4.4s}, [x1], #16\n\t"
"st1 {v6.4s}, [%[output_addr2]], #16\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"st1 {v5.4s}, [x1], #16\n\t"
"st1 {v7.4s}, [%[output_addr2]], #16\n\t"
"sub x2, %[block_num_w], #2\n\t"
"b int8_output_proc_dim4_loop_v6_%=\n\t"

"int8_output_proc_dim1_branch_v6_%=:\n\t"
"st1 {v4.d}[0], [x1], #8\n\t"
"st1 {v6.d}[0], [%[output_addr2]], #8\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"cmp %[block_num_w], #3\n\t"
"bge int8_output_proc_dim1_inner_dim3_branch_v6_%=\n\t"
"st1 {v4.d}[1], [x1], #8\n\t"
"st1 {v6.d}[1], [%[output_addr2]], #8\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"st1 {v5.d}[0], [x1], #8\n\t"
"st1 {v7.d}[0], [%[output_addr2]], #8\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"st1 {v5.d}[1], [x1], #8\n\t"
"st1 {v7.d}[1], [%[output_addr2]], #8\n\t"
"add x1, x1, %[output_offset]\n\t"
"add %[output_addr2], %[output_addr2], %[output_offset]\n\t"
"mov x2, %[block_num_w]\n\t"
"b int8_output_proc_dim4_loop_v6_%=\n\t"
"int8_output_proc_dim1_inner_dim3_branch_v6_%=:\n\t"
"st1 {v4.d}[1], [x1], #8\n\t"
"st1 {v6.d}[1], [%[output_addr2]], #8\n\t"
"mov x2, %[block_num_w]\n\t"
"st1 {v5.4s}, [x1], #16\n\t"
"st1 {v7.4s}, [%[output_addr2]], #16\n\t"
"sub x2, %[block_num_w], #3\n\t"
"b int8_output_proc_dim4_loop_v6_%=\n\t"

"int8_finish_output_proc_dim4_loop_v6_%=:\n\t"
: //output
: //input
[dim] "r" (block_num * ROUND(num_output, 4)),
[block_num_w] "r" (winograd_width / WINDOW_SIZE),
[input_addr] "r" (winograd_output),
[input_addr2] "r" (winograd_output + WINDOW_SIZE * WINDOW_SIZE),
[input_addr3] "r" (winograd_output + WINDOW_SIZE * WINDOW_SIZE * 2),
[input_addr4] "r" (winograd_output + WINDOW_SIZE * WINDOW_SIZE * 3),
[input_tri_offset] "r" (WINDOW_SIZE * WINDOW_SIZE * sizeof(int32_t) * 3),
[output_addr] "r" (winograd_pad_output),
[output_addr2] "r" (winograd_pad_output + winograd_width / WINDOW_SIZE * OUTPUT_SIZE),
[output_offset] "r" (winograd_width / WINDOW_SIZE * OUTPUT_SIZE * sizeof(int32_t))
: //clobbers
"cc", "memory", "x0", "x1", "x2",
"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
"v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
"v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
"v25", "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on

    size_t i, j;
    size_t winograd_pad_height = winograd_height / WINDOW_SIZE * OUTPUT_SIZE;
    size_t winograd_pad_width = winograd_width / WINDOW_SIZE * OUTPUT_SIZE;

    if (output_width != winograd_pad_width || output_height != winograd_pad_height) {
        for (i = 0; i < num_output; ++i) {
            for (j = 0; j < output_height; ++j) {
                memcpy_neon64(winograd_output, winograd_pad_output, sizeof(int32_t) * output_width);
                output += output_width;
                winograd_pad_output += winograd_pad_width;
            }
            winograd_pad_output += (winograd_pad_height - output_height) * winograd_pad_width;
        }
    }

    return;
}
#endif
