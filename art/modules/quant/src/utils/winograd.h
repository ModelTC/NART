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

#ifndef WINOGRAD_H
#define WINOGRAD_H

#define KERNEL_SIZE 3
#define WINDOW_SIZE 4
#define OUTPUT_SIZE 2
#define CONV_STRIDE 1
#define WINO_STRIDE 2

void winograd_k3s1_i8xi8(
    const int8_t *input, const int8_t *weights, mem_t *aux_mem1, mem_t *aux_mem2, mem_t *aux_mem3,
    mem_t *aux_mem4, int32_t *output, size_t channel, size_t height, size_t width,
    size_t num_output, size_t pad_h, size_t pad_w);
void winograd_input_process(
    int8_t *winograd_pad_input, int8_t *winograd_input, size_t winograd_pad_height,
    size_t winograd_pad_width, size_t channel);
void eltwise_mult_pseudo_gemm(
    int8_t *winograd_input, const int8_t *winograd_weights, int32_t *winograd_output,
    size_t winograd_height, size_t winograd_width, size_t channel, size_t num_output);
void winograd_output_process(
    int32_t *winograd_output, int32_t *winograd_pad_output, int32_t *output, size_t output_height,
    size_t output_width, size_t winograd_height, size_t winograd_width, size_t num_output);

#endif
