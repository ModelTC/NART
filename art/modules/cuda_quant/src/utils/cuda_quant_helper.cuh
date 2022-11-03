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

#ifndef CUDA_QUANT_HELPER_H
#define CUDA_QUANT_HELPER_H

#include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include "../cuda_quant_workspace.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif
void round_clip_cu(
    const float *input, int8_t *output, size_t num, int l, int h, float alpha, workspace_t *ws);
void saturate_float_by_bits_cu(
    const float *input, uint8_t *output, uint8_t bits, size_t num, float alpha, float beta,
    workspace_t *ws);
void saturate_int_by_bits_cu(
    int32_t *input, uint8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift,
    uint8_t bits, workspace_t *ws);
void ssaturate_int_by_bits_cu(
    int32_t *input, int8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift, uint8_t bits,
    workspace_t *ws);

void dequantize_u8_cu(
    const uint8_t *input, float *output, size_t num, uint8_t zero_point, float alpha,
    workspace_t *ws);
void dequantize_i8_cu(const int8_t *input, float *output, size_t num, float alpha, workspace_t *ws);
void dequantize_i4_cu(const int8_t *input, float *output, size_t num, float alpha, workspace_t *ws);

void get_quant_scale_int32(float x, uint32_t *multer, int8_t *shift);
void ssaturate_int_by_bits_cu_temp(
    int32_t *input, int8_t *output, size_t num, size_t row_pad, size_t row_actual, size_t col_pad,
    size_t col_actual, uint32_t res_mult, uint32_t res_shift, uint8_t bits, workspace_t *ws);

// void conv_tran_cnhw_2_nchw_fuse_bias_sat_cu(int* in, int* bias, int* out, int batch, int channel,
// int o_w, int o_h, int col_pad, int row_pad, uint32_t res_mult, uint32_t res_shift, uint8_t bits,
// bool has_bias, workspace_t * ws);

void transpose_NCHW_2_NHWC_quantize_i8_cu(
    float *input, int8_t *output, int n, int c, int h, int w, float alpha, workspace_t *ws);
void transpose_NHWC_2_NCHW_dequantize_i8_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws);
void transpose_NCHW_2_NHWC_quantize_i4_cu(
    float *input, int8_t *output, int n, int c, int h, int w, float alpha, workspace_t *ws);
void transpose_NHWC_2_NCHW_dequantize_i4_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws);

#ifdef __cplusplus
}
#endif

#endif
