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

#ifndef GEMM_H
#define GEMM_H

#include "art/mem.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifdef _WIN32
#define OPTIMIZE_O0
#else
#define OPTIMIZE_O0 __attribute__((optimize("O0")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t gemm_i2xi2_auxmem_size(int M, int N, int K);
size_t gemm_i3xi3_auxmem_size(int M, int N, int K);
size_t gemm_i4xi4_auxmem_size(int M, int N, int K);
size_t gemm_i6xi6_auxmem_size(int M, int N, int K);
size_t gemm_i7xi7_auxmem_size(int M, int N, int K);
size_t gemm_i8xi8_auxmem_size(int M, int N, int K);
size_t gemm_i16xi16_auxmem_size(int M, int N, int K);

void gemm_i2xi2(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i3xi3(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i4xi4(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i6xi6(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i7xi7(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i8xi8(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem) OPTIMIZE_O0;
void gemm_i16xi16(
    const size_t M, const size_t N, const size_t K, const int16_t *A, const int16_t *B, int32_t *C,
    mem_t *aux_mem);
void gemm_ATxB_i16xi16(
    const size_t M, const size_t N, const size_t K, const int16_t *A, const int16_t *B, int32_t *C,
    mem_t *aux_mem_1, mem_t *aux_mem_2);

#ifdef __cplusplus
}
#endif

#endif
