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

#ifndef IM2COL_H
#define IM2COL_H

#ifdef __cplusplus
extern "C" {
#endif

void im2col_i8_default(
    const int8_t *data, int8_t *col, const size_t channel, const size_t height, const size_t width,
    const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h, const uint16_t pad_w,
    const uint16_t stride_h, const uint16_t stride_w);

void im2col_i8(
    const int8_t *data, int8_t *col, const size_t channel, const size_t height, const size_t width,
    const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h, const uint16_t pad_w,
    const uint16_t stride_h, const uint16_t stride_w);

void im2col_i16(
    const int16_t *data, int16_t *col, const size_t channel, const size_t height,
    const size_t width, const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h,
    const uint16_t pad_w, const uint16_t stride_h, const uint16_t stride_w);

void col2im_i32(
    const int32_t *col, int32_t *data, const size_t channel, const size_t height,
    const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
    const size_t pad_w, const size_t stride_h, const size_t stride_w);

#ifdef __cplusplus
}
#endif

#endif
