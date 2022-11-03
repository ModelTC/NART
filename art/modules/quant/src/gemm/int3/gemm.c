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

size_t gemm_i3xi3_auxmem_size(int M, int N, int K)
{
    size_t sz = ((M + 4 - 1) / 4 * 4 * K) * sizeof(int8_t) + // packing_col_padding_A
        ((N + 16 - 1) / 16 * 16 * K) * sizeof(int8_t) + // packing_col_padding_B
        ((M + 4 - 1) / 4) * 4 * ((N + 16 - 1) / 16) * 16 * sizeof(int32_t); // padd_C

    return sz;
}

void gemm_i3xi3(
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
"int3_dim_m4x16_loop_%=:\n\t"
"cmp %[dim_m], #4\n\t"
"blt int3_finish_dim_m4x16_loop_%=\n\t"
"mov x8, %[dim_n]\n\t"
"mov x9, %[addr_b]\n\t"

"int3_dim_n4x16_loop_%=:\n\t"
"cmp x8, #16\n\t"
"blt int3_finish_dim_n4x16_loop_%=\n\t"
"mov x10, %[addr_a]\n\t"
"mov x11, %[dim_k]\n\t"

"mov x0, #0\n\t"
"movi v12.2d, #0\n\t"
"movi v13.2d, #0\n\t"
"mov x1, #0\n\t"
"movi v14.2d, #0\n\t"
"movi v15.2d, #0\n\t"
"mov x2, #0\n\t"
"movi v16.2d, #0\n\t"
"movi v17.2d, #0\n\t"
"mov x3, #0\n\t"
"movi v18.2d, #0\n\t"
"movi v19.2d, #0\n\t"
"mov x4, #0\n\t"
"movi v20.2d, #0\n\t"
"movi v21.2d, #0\n\t"
"mov x5, #0\n\t"
"movi v22.2d, #0\n\t"
"movi v23.2d, #0\n\t"
"mov x6, #0\n\t"
"movi v24.2d, #0\n\t"
"movi v25.2d, #0\n\t"
"mov x7, #0\n\t"
"movi v26.2d, #0\n\t"
"movi v27.2d, #0\n\t"
"movi v28.2d, #0\n\t"
"movi v29.2d, #0\n\t"
"movi v30.2d, #0\n\t"
"movi v31.2d, #0\n\t"

"int3_dim_k4x16x252_loop_%=:\n\t"
"cmp x11, #252\n\t"
"blt int3_finish_dim_k4x16x252_loop_%=\n\t"
"mov x12, #36\n\t"

"int3_dim_k4x16x252_16232_loop_%=:\n\t"
"cmp x12, #0\n\t"
"beq int3_finish_dim_k4x16x252_16232_loop_%=\n\t"
// "mov x13, #3\n\t"

"ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x9], #64\n\t"
"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mul v8.16b, v0.16b, v4.16b\n\t"
"mul v9.16b, v1.16b, v4.16b\n\t"
"mul v10.16b, v2.16b, v4.16b\n\t"
"mul v11.16b, v3.16b, v4.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v5.16b\n\t"
"mla v9.16b, v1.16b, v5.16b\n\t"
"mla v10.16b, v2.16b, v5.16b\n\t"
"mla v11.16b, v3.16b, v5.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v6.16b\n\t"
"mla v9.16b, v1.16b, v6.16b\n\t"
"mla v10.16b, v2.16b, v6.16b\n\t"
"mla v11.16b, v3.16b, v6.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v7.16b\n\t"
"mla v9.16b, v1.16b, v7.16b\n\t"
"mla v10.16b, v2.16b, v7.16b\n\t"
"mla v11.16b, v3.16b, v7.16b\n\t"

// "int3_dim_k4x16x252_8216_loop_%=:\n\t"
// "cmp x13, #0\n\t"
// "beq int3_finish_dim_k4x16x252_8216_loop_%=\n\t"

"ld1 {v4.16b, v5.16b, v6.16b}, [x9], #48\n\t"
"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v4.16b\n\t"
"mla v9.16b, v1.16b, v4.16b\n\t"
"mla v10.16b, v2.16b, v4.16b\n\t"
"mla v11.16b, v3.16b, v4.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v5.16b\n\t"
"mla v9.16b, v1.16b, v5.16b\n\t"
"mla v10.16b, v2.16b, v5.16b\n\t"
"mla v11.16b, v3.16b, v5.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v6.16b\n\t"
"mla v9.16b, v1.16b, v6.16b\n\t"
"mla v10.16b, v2.16b, v6.16b\n\t"
"mla v11.16b, v3.16b, v6.16b\n\t"

// "sub x13, x13, #1\n\t"
// "b int3_dim_k4x16x252_8216_loop_%=\n\t"
// "int3_finish_dim_k4x16x252_8216_loop_%=:\n\t"

"saddw  v12.8h, v12.8h, v8.8b\n\t"
"saddw2 v13.8h, v13.8h, v8.16b\n\t"
"saddw  v14.8h, v14.8h, v9.8b\n\t"
"saddw2 v15.8h, v15.8h, v9.16b\n\t"
"sub x12, x12, #1\n\t"
"saddw  v16.8h, v16.8h, v10.8b\n\t"
"saddw2 v17.8h, v17.8h, v10.16b\n\t"
"saddw  v18.8h, v18.8h, v11.8b\n\t"
"saddw2 v19.8h, v19.8h, v11.16b\n\t"

"b int3_dim_k4x16x252_16232_loop_%=\n\t"
"int3_finish_dim_k4x16x252_16232_loop_%=:\n\t"

"saddw  v20.4s, v20.4s, v12.4h\n\t"
"saddw2 v21.4s, v21.4s, v12.8h\n\t"
"mov v8.d[0], x0\n\t"
"saddw  v22.4s, v22.4s, v13.4h\n\t"
"saddw2 v23.4s, v23.4s, v13.8h\n\t"
"mov v8.d[1], x1\n\t"
"saddw  v24.4s, v24.4s, v14.4h\n\t"
"saddw2 v25.4s, v25.4s, v14.8h\n\t"
"mov v9.d[0], x2\n\t"
"saddw  v26.4s, v26.4s, v15.4h\n\t"
"saddw2 v27.4s, v27.4s, v15.8h\n\t"
"mov v9.d[1], x3\n\t"
"saddw  v28.4s, v28.4s, v16.4h\n\t"
"saddw2 v29.4s, v29.4s, v16.8h\n\t"
"mov v10.d[0], x4\n\t"
"saddw  v30.4s, v30.4s, v17.4h\n\t"
"saddw2 v31.4s, v31.4s, v17.8h\n\t"
"mov v10.d[1], x5\n\t"
"mov v11.d[0], x6\n\t"
"mov v11.d[1], x7\n\t"
"movi v12.2d, #0\n\t"
"movi v13.2d, #0\n\t"
"movi v14.2d, #0\n\t"
"movi v15.2d, #0\n\t"
"saddw  v8.4s, v8.4s, v18.4h\n\t"
"saddw2 v9.4s, v9.4s, v18.8h\n\t"
"movi v16.2d, #0\n\t"
"movi v17.2d, #0\n\t"
"saddw  v10.4s, v10.4s, v19.4h\n\t"
"saddw2 v11.4s, v11.4s, v19.8h\n\t"
"sub x11, x11, #252\n\t"
"mov x0, v8.d[0]\n\t"
"mov x1, v8.d[1]\n\t"
"movi v18.2d, #0\n\t"
"mov x2, v9.d[0]\n\t"
"mov x3, v9.d[1]\n\t"
"mov x4, v10.d[0]\n\t"
"mov x5, v10.d[1]\n\t"
"movi v19.2d, #0\n\t"
"mov x6, v11.d[0]\n\t"
"mov x7, v11.d[1]\n\t"

"b int3_dim_k4x16x252_loop_%=\n\t"
"int3_finish_dim_k4x16x252_loop_%=:\n\t"

"int3_dim_k4x16x8_loop_%=:\n\t"
"cmp x11, #8\n\t"
"blt int3_finish_dim_k4x16x8_loop_%=\n\t"

"ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x9], #64\n\t"
"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mul v8.16b, v0.16b, v4.16b\n\t"
"mul v9.16b, v1.16b, v4.16b\n\t"
"mul v10.16b, v2.16b, v4.16b\n\t"
"mul v11.16b, v3.16b, v4.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v5.16b\n\t"
"mla v9.16b, v1.16b, v5.16b\n\t"
"mla v10.16b, v2.16b, v5.16b\n\t"
"mla v11.16b, v3.16b, v5.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v6.16b\n\t"
"mla v9.16b, v1.16b, v6.16b\n\t"
"mla v10.16b, v2.16b, v6.16b\n\t"
"mla v11.16b, v3.16b, v6.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v7.16b\n\t"
"mla v9.16b, v1.16b, v7.16b\n\t"
"mla v10.16b, v2.16b, v7.16b\n\t"
"mla v11.16b, v3.16b, v7.16b\n\t"

"ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x9], #64\n\t"
"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v4.16b\n\t"
"mla v9.16b, v1.16b, v4.16b\n\t"
"mla v10.16b, v2.16b, v4.16b\n\t"
"mla v11.16b, v3.16b, v4.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v5.16b\n\t"
"mla v9.16b, v1.16b, v5.16b\n\t"
"mla v10.16b, v2.16b, v5.16b\n\t"
"mla v11.16b, v3.16b, v5.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v6.16b\n\t"
"mla v9.16b, v1.16b, v6.16b\n\t"
"mla v10.16b, v2.16b, v6.16b\n\t"
"mla v11.16b, v3.16b, v6.16b\n\t"

"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t" // load matrix a

"mla v8.16b, v0.16b, v7.16b\n\t"
"mla v9.16b, v1.16b, v7.16b\n\t"
"mla v10.16b, v2.16b, v7.16b\n\t"
"mla v11.16b, v3.16b, v7.16b\n\t"

"sub x11, x11, #8\n\t"

"saddw  v12.8h, v12.8h, v8.8b\n\t"
"saddw2 v13.8h, v13.8h, v8.16b\n\t"
"saddw  v14.8h, v14.8h, v9.8b\n\t"
"saddw2 v15.8h, v15.8h, v9.16b\n\t"
"saddw  v16.8h, v16.8h, v10.8b\n\t"
"saddw2 v17.8h, v17.8h, v10.16b\n\t"
"saddw  v18.8h, v18.8h, v11.8b\n\t"
"saddw2 v19.8h, v19.8h, v11.16b\n\t"

"b int3_dim_k4x16x8_loop_%=\n\t"
"int3_finish_dim_k4x16x8_loop_%=:\n\t"

"int3_dim_k4x16x1_loop_%=:\n\t"
"cmp x11, #0\n\t"
"beq int3_finish_dim_k4x16x1_loop_%=\n\t"

"ld1 {v4.2d}, [x9], #16\n\t"
"ld4r {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #4\n\t"

"smlal  v12.8h, v0.8b,  v4.8b\n\t"
"smlal2 v13.8h, v0.16b, v4.16b\n\t"
"smlal  v14.8h, v1.8b,  v4.8b\n\t"
"smlal2 v15.8h, v1.16b, v4.16b\n\t"
"sub x11, x11, #1\n\t"
"smlal  v16.8h, v2.8b,  v4.8b\n\t"
"smlal2 v17.8h, v2.16b, v4.16b\n\t"
"smlal  v18.8h, v3.8b,  v4.8b\n\t"
"smlal2 v19.8h, v3.16b, v4.16b\n\t"

"b int3_dim_k4x16x1_loop_%=\n\t"
"int3_finish_dim_k4x16x1_loop_%=:\n\t"

"saddw  v20.4s, v20.4s, v12.4h\n\t"
"saddw2 v21.4s, v21.4s, v12.8h\n\t"
"mov v8.d[0], x0\n\t"
"saddw  v22.4s, v22.4s, v13.4h\n\t"
"saddw2 v23.4s, v23.4s, v13.8h\n\t"
"mov v8.d[1], x1\n\t"
"saddw  v24.4s, v24.4s, v14.4h\n\t"
"saddw2 v25.4s, v25.4s, v14.8h\n\t"
"mov v9.d[0], x2\n\t"
"saddw  v26.4s, v26.4s, v15.4h\n\t"
"saddw2 v27.4s, v27.4s, v15.8h\n\t"
"mov v9.d[1], x3\n\t"
"saddw  v28.4s, v28.4s, v16.4h\n\t"
"saddw2 v29.4s, v29.4s, v16.8h\n\t"
"mov v10.d[0], x4\n\t"
"saddw  v30.4s, v30.4s, v17.4h\n\t"
"saddw2 v31.4s, v31.4s, v17.8h\n\t"
"mov v10.d[1], x5\n\t"
"mov v11.d[0], x6\n\t"
"mov v11.d[1], x7\n\t"
"saddw  v8.4s, v8.4s, v18.4h\n\t"
"saddw2 v9.4s, v9.4s, v18.8h\n\t"
"saddw  v10.4s, v10.4s, v19.4h\n\t"
"saddw2 v11.4s, v11.4s, v19.8h\n\t"

"st1 {v20.2d, v21.2d, v22.2d, v23.2d}, [%[addr_c]], %[n_offset]\n\t"
"st1 {v24.2d, v25.2d, v26.2d, v27.2d}, [%[addr_c]], %[n_offset]\n\t"
"sub x8, x8, #16\n\t"
"st1 {v28.2d, v29.2d, v30.2d, v31.2d}, [%[addr_c]], %[n_offset]\n\t"
"st1 {v8.2d, v9.2d, v10.2d, v11.2d}, [%[addr_c]], #64\n\t"
"sub %[addr_c], %[addr_c], %[resume_n_offset]\n\t"

"b int3_dim_n4x16_loop_%=\n\t"
"int3_finish_dim_n4x16_loop_%=:\n\t"

"sub %[dim_m], %[dim_m], #4\n\t"
"add %[addr_a], %[addr_a], %[a_offset]\n\t"
"add %[addr_c], %[addr_c], %[resume_n_offset]\n\t"
"b int3_dim_m4x16_loop_%=\n\t"
"int3_finish_dim_m4x16_loop_%=:\n\t"
: // output
: // input
[dim_m] "r" (ROUND(M, 4)), [dim_n] "r" (ROUND(N, 16)), [dim_k] "r" (K),
[addr_a] "r" (pack_A), [addr_b] "r" (pack_B), [addr_c] "r" (padd_C),
[a_offset] "r" (a_offset), [n_offset] "r" (n_offset),
[resume_n_offset] "r" (resume_n_offset)
: //clobber
"cc", "memory" ,
"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
"x9", "x10", "x11", "x12",
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
