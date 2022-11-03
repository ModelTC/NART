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
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/timer.h"

#include "../../utils/utils.h"
#include "../gemm.h"

size_t gemm_i6xi6_auxmem_size(int M, int N, int K)
{
    size_t sz = ((M + 4 - 1) / 4 * 4 * K) * sizeof(int8_t) + // packing_col_padding_A
        ((N + 16 - 1) / 16 * 16 * K) * sizeof(int8_t) + // packing_col_padding_B
        ((M + 4 - 1) / 4) * 4 * ((N + 16 - 1) / 16) * 16 * sizeof(int32_t); // padd_C

    return sz;
}

void gemm_i6xi6(
    const size_t M, const size_t N, const size_t K, const int8_t *A, const int8_t *B, int32_t *C,
    mem_t *aux_mem)
{
#if defined(__aarch64__) && !defined(__APPLE__)
    size_t a_offset = K * sizeof(int8_t) * 4;
    size_t n_offset = ROUND(N, 16) * sizeof(int32_t);
    size_t resume_n_offset = n_offset * (4 - 1);
    void *ws = mem_cpu_data(aux_mem);

    size_t sz = ROUND(M, 4) * K * sizeof(int8_t) + ROUND(N, 16) * K * sizeof(int8_t)
        + ROUND(M, 4) * ROUND(N, 16) * sizeof(int32_t);
    CHECK_GE(mem_sizeof(aux_mem), sz);

    TI(packing);
    int8_t *pack_A = ws;
    // required size: (M + 4 - 1) / 4 * K
    ws = packing_col_padding_int8(A, M, K, 4, ws);
    // int8_t* pack_B = packing_row_padding_int8(B, K, N, 16);
    int8_t *pack_B = ws;
    // required size: (N + 16 - 1) / 4 * K
    ws = packing_col_padding_int8(B, N, K, 16, ws);
    int32_t *padd_C = ws;
    // required size: ((M  + 4 - 1) / 4) * ((N + 16 - 1) / 4)
    TI(memset);
    // memset(padd_C, 0, sizeof(int32_t) * ROUND(M, 4) * ROUND(N, 16));
    TO(memset, "memset_padd_C ");
    TO(packing, "packing ");

    // clang-format off
__asm__ __volatile__(
"int6_dim_m4x16_loop_%=:\n\t"
"cmp %[dim_m], #4\n\t"
"blt int6_finish_dim_m4x16_loop_%=\n\t"
"mov x4, %[dim_n]\n\t"
"mov x5, %[addr_b]\n\t"

"int6_dim_n4x16_loop_%=:\n\t"
"cmp x4, #16\n\t"
"blt int6_finish_dim_n4x16_loop_%=\n\t"
"mov x6, %[addr_a]\n\t"
"mov x7, %[dim_k]\n\t"

"movi v18.2d, #0\n\t"
"movi v19.2d, #0\n\t"
"mov  x0, #0\n\t"
"movi v20.2d, #0\n\t"
"movi v21.2d, #0\n\t"
"mov  x1, #0\n\t"
"movi v22.2d, #0\n\t"
"movi v23.2d, #0\n\t"
"mov  x2, #0\n\t"
"movi v24.2d, #0\n\t"
"movi v25.2d, #0\n\t"
"mov  x3, #0\n\t"
"movi v26.2d, #0\n\t"
"movi v27.2d, #0\n\t"
"movi v28.2d, #0\n\t"
"movi v29.2d, #0\n\t"
"movi v30.2d, #0\n\t"
"movi v31.2d, #0\n\t"

"int6_dim_k4x16x16_loop_%=:\n\t"
"cmp x7, #16\n\t"
"blt int6_finish_dim_k4x16x16_loop_%=\n\t"

"ld1 {v0.2d, v1.2d}, [x5], #32\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smull  v10.8h, v0.8b,  v2.8b\n\t"
"smull2 v11.8h, v0.16b, v2.16b\n\t"
"smull  v12.8h, v0.8b,  v3.8b\n\t"
"smull2 v13.8h, v0.16b, v3.16b\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smull  v14.8h, v0.8b,  v4.8b\n\t"
"smull2 v15.8h, v0.16b, v4.16b\n\t"
"smull  v16.8h, v0.8b,  v5.8b\n\t"
"smull2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v1.8b,  v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"ld1 {v1.2d}, [x5], #16\n\t"
"ld4r {v6.16b, v7.16b, v8.16b, v9.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"smlal  v10.8h, v1.8b, v6.8b\n\t"
"smlal2 v11.8h, v1.16b, v6.16b\n\t"
"smlal  v12.8h, v1.8b,  v7.8b\n\t"
"smlal2 v13.8h, v1.16b, v7.16b\n\t"
"smlal  v14.8h, v1.8b,  v8.8b\n\t"
"smlal2 v15.8h, v1.16b, v8.16b\n\t"
"smlal  v16.8h, v1.8b,  v9.8b\n\t"
"smlal2 v17.8h, v1.16b, v9.16b\n\t"

"mov v0.d[0], x0\n\t"
"saddw  v18.4s, v18.4s, v10.4h\n\t"
"saddw2 v19.4s, v19.4s, v10.8h\n\t"
"mov v0.d[1], x1\n\t"
"saddw  v20.4s, v20.4s, v11.4h\n\t"
"saddw2 v21.4s, v21.4s, v11.8h\n\t"
"mov v1.d[0], x2\n\t"
"saddw  v22.4s, v22.4s, v12.4h\n\t"
"saddw2 v23.4s, v23.4s, v12.8h\n\t"
"mov v1.d[1], x3\n\t"
"saddw  v24.4s, v24.4s, v13.4h\n\t"
"saddw2 v25.4s, v25.4s, v13.8h\n\t"
"saddw  v26.4s, v26.4s, v14.4h\n\t"
"saddw2 v27.4s, v27.4s, v14.8h\n\t"
"saddw  v28.4s, v28.4s, v15.4h\n\t"
"saddw2 v29.4s, v29.4s, v15.8h\n\t"
"sub x7, x7, #16\n\t"
"saddw  v30.4s, v30.4s, v16.4h\n\t"
"saddw2 v31.4s, v31.4s, v16.8h\n\t"
"saddw  v0.4s, v0.4s, v17.4h\n\t"
"saddw2 v1.4s, v1.4s, v17.8h\n\t"
"mov x0, v0.d[0]\n\t"
"mov x1, v0.d[1]\n\t"
"mov x2, v1.d[0]\n\t"
"mov x3, v1.d[1]\n\t"

"b int6_dim_k4x16x16_loop_%=\n\t"
"int6_finish_dim_k4x16x16_loop_%=:\n\t"

"movi v10.2d, #0\n\t"
"movi v11.2d, #0\n\t"
"movi v12.2d, #0\n\t"
"movi v13.2d, #0\n\t"
"movi v14.2d, #0\n\t"
"movi v15.2d, #0\n\t"
"movi v16.2d, #0\n\t"
"movi v17.2d, #0\n\t"

"int6_dim_k4x16x1_loop_%=:\n\t"
"cmp x7, #0\n\t"
"beq int6_finish_dim_k4x16x1_loop_%=\n\t"

"ld1 {v0.2d}, [x5], #16\n\t"
"ld4r {v2.16b, v3.16b, v4.16b, v5.16b}, [x6], #4\n\t"
"smlal  v10.8h, v0.8b,  v2.8b\n\t"
"smlal2 v11.8h, v0.16b, v2.16b\n\t"
"smlal  v12.8h, v0.8b,  v3.8b\n\t"
"smlal2 v13.8h, v0.16b, v3.16b\n\t"
"sub x7, x7, #1\n\t"
"smlal  v14.8h, v0.8b,  v4.8b\n\t"
"smlal2 v15.8h, v0.16b, v4.16b\n\t"
"smlal  v16.8h, v0.8b,  v5.8b\n\t"
"smlal2 v17.8h, v0.16b, v5.16b\n\t"

"b int6_dim_k4x16x1_loop_%=\n\t"
"int6_finish_dim_k4x16x1_loop_%=:\n\t"

"mov v0.d[0], x0\n\t"
"saddw  v18.4s, v18.4s, v10.4h\n\t"
"saddw2 v19.4s, v19.4s, v10.8h\n\t"
"mov v0.d[1], x1\n\t"
"saddw  v20.4s, v20.4s, v11.4h\n\t"
"saddw2 v21.4s, v21.4s, v11.8h\n\t"
"mov v1.d[0], x2\n\t"
"saddw  v22.4s, v22.4s, v12.4h\n\t"
"saddw2 v23.4s, v23.4s, v12.8h\n\t"
"mov v1.d[1], x3\n\t"
"saddw  v24.4s, v24.4s, v13.4h\n\t"
"saddw2 v25.4s, v25.4s, v13.8h\n\t"
"saddw  v26.4s, v26.4s, v14.4h\n\t"
"saddw2 v27.4s, v27.4s, v14.8h\n\t"
"saddw  v28.4s, v28.4s, v15.4h\n\t"
"saddw2 v29.4s, v29.4s, v15.8h\n\t"
"saddw  v30.4s, v30.4s, v16.4h\n\t"
"saddw2 v31.4s, v31.4s, v16.8h\n\t"
"saddw  v0.4s, v0.4s, v17.4h\n\t"
"saddw2 v1.4s, v1.4s, v17.8h\n\t"
// "mov x0, v0.d[0]\n\t"
// "mov x1, v0.d[1]\n\t"
// "mov x2, v1.d[0]\n\t"
// "mov x3, v1.d[1]\n\t"

"st1 {v18.4s, v19.4s, v20.4s, v21.4s}, [%[addr_c]], %[n_offset]\n\t"
// "mov v0.d[0], x0\n\t"
// "mov v0.d[1], x1\n\t"
"st1 {v22.4s, v23.4s, v24.4s, v25.4s}, [%[addr_c]], %[n_offset]\n\t"
// "mov v1.d[0], x2\n\t"
// "mov v1.d[1], x3\n\t"
"st1 {v26.4s, v27.4s, v28.4s, v29.4s}, [%[addr_c]], %[n_offset]\n\t"
"sub x4, x4, #16\n\t"
"st1 {v30.4s, v31.4s}, [%[addr_c]], #32\n\t"
"st1 {v0.4s, v1.4s}, [%[addr_c]], #32\n\t"
"sub %[addr_c], %[addr_c], %[resume_n_offset]\n\t"

"b int6_dim_n4x16_loop_%=\n\t"
"int6_finish_dim_n4x16_loop_%=:\n\t"

"sub %[dim_m], %[dim_m], #4\n\t"
"add %[addr_a], %[addr_a], %[a_offset]\n\t"
"add %[addr_c], %[addr_c], %[resume_n_offset]\n\t"
"b int6_dim_m4x16_loop_%=\n\t"
"int6_finish_dim_m4x16_loop_%=:\n\t"
: // output
: // input
[dim_m] "r" (ROUND(M, 4)), [dim_n] "r" (ROUND(N, 16)), [dim_k] "r" (K),
[addr_a] "r" (pack_A), [addr_b] "r" (pack_B), [addr_c] "r" (padd_C),
[a_offset] "r" (a_offset), [n_offset] "r" (n_offset),
[resume_n_offset] "r" (resume_n_offset)
: //clobber
"cc", "memory" , "x0", "x1", "x2", "x3", "x4", "x5",
"x6", "x7", "x8",
"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
"v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
"v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
"v25", "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on

    size_t i;
    for (i = 0; i < M; ++i) {
        memcpy(C, padd_C, sizeof(int32_t) * N);
        C += N;
        padd_C += ROUND(N, 16);
    }
    padd_C -= M * ROUND(N, 16);
#else
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)B;
    (void)C;
    (void)aux_mem;
    LOG(error, "%s Not Implemented\n", __func__);
#endif
}
