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

#ifndef UTILS_H
#define UTILS_H
// #include "art/art/mem.h"

#include "art/mem.h"

#define ROUND(a, b) (((a) + (b)-1) / (b) * (b))

void transpose_i8(const int8_t *data, mem_t *aux_mem, size_t M, size_t N);
void transpose_i16(const int16_t *data, mem_t *aux_mem, size_t M, size_t N);
void transpose_inplace_i32(int32_t *data, mem_t *aux_mem, size_t M, size_t N);

int8_t *packing_col_int8(const int8_t *data, size_t M, size_t N, size_t m);
int8_t *packing_row_int8(const int8_t *data, size_t M, size_t N, size_t n);
int8_t *packing_col_padding_int8(const int8_t *data, size_t M, size_t N, size_t m, int8_t *aux);
int8_t *packing_row_padding_int8(const int8_t *data, size_t M, size_t N, size_t n);
void depacking_col_inplace_int8(int8_t *data_pack, size_t M, size_t N, size_t m);
void depacking_col_inplace_int32(int32_t *data_pack, size_t M, size_t N, size_t m);

void memcpy_neon64(void *dst, const void *src, size_t count);

#endif
