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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/timer.h"

#include "utils.h"

#include "../gemm/gemm.h"

// 调用完成后注意释放临时申请的空间
void transpose_i8(const int8_t *data, mem_t *aux_mem, size_t M, size_t N)
{
    size_t i, j;
    size_t count = M * N;
    int8_t *data_trans = mem_cpu_data(aux_mem);

    for (i = N; i > 0; --i) {
        for (j = M; j > 0; --j) {
            *data_trans++ = *data;
            data += N;
        }
        data -= count - 1;
    }
}

void transpose_i16(const int16_t *data, mem_t *aux_mem, size_t M, size_t N)
{
    size_t i, j;
    size_t count = M * N;
    int16_t *data_trans = mem_cpu_data(aux_mem);

    for (i = N; i > 0; --i) {
        for (j = M; j > 0; --j) {
            *data_trans++ = *data;
            data += N;
        }
        data -= count - 1;
    }
}

void transpose_inplace_i32(int32_t *data, mem_t *aux_mem, size_t M, size_t N)
{
    size_t i, j;
    size_t count = M * N;
    int32_t *data_trans = mem_cpu_data(aux_mem);

    for (i = N; i > 0; --i) {
        for (j = M; j > 0; --j) {
            *data_trans++ = *data;
            data += N;
        }
        data -= count - 1;
    }
    data -= N;
    data_trans -= count;

    memcpy(data, data_trans, sizeof(int32_t) * count);
}

int8_t *packing_col_int8(const int8_t *data, size_t M, size_t N, size_t m)
{
    size_t i, j, k;
    size_t pack_m = ROUND(M, m);
    size_t pack_size = N * m;

    int8_t *data_pack = (int8_t *)malloc(sizeof(int8_t) * M * N);
    memset(data_pack, 0, sizeof(int8_t) * M * N);

    for (i = 0; i < M / m; ++i) {
        for (j = 0; j < m; ++j) {
            for (k = 0; k < N; ++k) {
                *data_pack = *data++;
                data_pack += m;
            }
            data_pack -= pack_size - 1;
        }
        data_pack += pack_size - m;
    }

    if (0 != M % m) {
        size_t remain_m = m - (pack_m - M);
        size_t remain_pack_size = N * remain_m;
        for (i = 0; i < remain_m; ++i) {
            for (j = 0; j < N; ++j) {
                *data_pack = *data++;
                data_pack += remain_m;
            }
            data_pack -= remain_pack_size - 1;
        }
        data_pack += remain_pack_size - remain_m;
    }

    return data_pack - M * N;
}

int8_t *packing_col_padding_int8(const int8_t *data, size_t M, size_t N, size_t m, int8_t *aux)
{
    size_t i, j, k;
    size_t pack_m = ROUND(M, m);
    size_t pack_size = N * m;

    int8_t *data_pack = aux;
    // memset(data_pack, 0, sizeof(int8_t) * pack_m * N);
    TI(pure_pack);

    if (4 == m) {
        for (i = 0; i < pack_m / m; ++i) {
            for (j = 0; j < N; ++j) {
                *data_pack++ = data[0 * N];
                *data_pack++ = data[1 * N];
                *data_pack++ = data[2 * N];
                *data_pack++ = data[3 * N];
                data++;
            }
            data += (m - 1) * N;
        }
    } else if (16 == m) {
        for (i = 0; i < pack_m / m; ++i) {
            for (j = 0; j < N; ++j) {
                *data_pack++ = data[0 * N];
                *data_pack++ = data[1 * N];
                *data_pack++ = data[2 * N];
                *data_pack++ = data[3 * N];
                *data_pack++ = data[4 * N];
                *data_pack++ = data[5 * N];
                *data_pack++ = data[6 * N];
                *data_pack++ = data[7 * N];
                *data_pack++ = data[8 * N];
                *data_pack++ = data[9 * N];
                *data_pack++ = data[10 * N];
                *data_pack++ = data[11 * N];
                *data_pack++ = data[12 * N];
                *data_pack++ = data[13 * N];
                *data_pack++ = data[14 * N];
                *data_pack++ = data[15 * N];
                data++;
            }
            data += (m - 1) * N;
        }
    } else {
        for (i = 0; i < pack_m / m; ++i) {
            for (j = 0; j < m; ++j) {
                for (k = 0; k < N; ++k) {
                    *data_pack = *data++;
                    data_pack += m;
                }
                data_pack -= pack_size - 1;
            }
            data_pack += pack_size - m;
        }
    }

    TO(pure_pack, "pure_pack ");
    return data_pack;
}

int8_t *packing_row_int8(const int8_t *data, size_t M, size_t N, size_t n)
{
    size_t i, j;
    size_t pack_n = ROUND(N, n);
    size_t copy_size = sizeof(int8_t) * n;
    size_t pack_size = M * n;

    int8_t *data_pack = (int8_t *)malloc(sizeof(int8_t) * M * N);
    memset(data_pack, 0, sizeof(int8_t) * M * N);

    if (0 == N % n) {
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N / n; ++j) {
                memcpy(data_pack, data, copy_size);
                data += n;
                data_pack += pack_size;
            }
            data_pack -= M * N - n;
        }
    } else {
        size_t remain_n = n - (pack_n - N);
        int8_t *temp = NULL;
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N / n; ++j) {
                memcpy(data_pack, data, copy_size);
                data += n;
                data_pack += pack_size;
            }
            temp = data_pack - remain_n * i;
            for (j = 0; j < remain_n; ++j) {
                temp[j] = *data++;
            }
            data_pack -= (N - remain_n) * M;
        }
    }

    return data_pack - pack_size;
}

int8_t *packing_row_padding_int8(const int8_t *data, size_t M, size_t N, size_t n)
{
    size_t i, j;
    size_t pack_n = ROUND(N, n);
    size_t copy_size = sizeof(int8_t) * n;
    size_t pack_size = M * n;
    size_t data_pack_count = M * pack_n;

    int8_t *data_pack = (int8_t *)malloc(sizeof(int8_t) * M * pack_n);
    memset(data_pack, 0, sizeof(int8_t) * M * pack_n);

    if (0 == N % n) {
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N / n; ++j) {
                memcpy(data_pack, data, copy_size);
                data += n;
                data_pack += pack_size;
            }
            data_pack -= data_pack_count - n;
        }
    } else {
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N / n; ++j) {
                memcpy(data_pack, data, copy_size);
                data += n;
                data_pack += pack_size;
            }
            for (j = 0; j < n - (pack_n - N); ++j) {
                data_pack[j] = *data++;
            }
            data_pack -= data_pack_count - n - pack_size;
        }
    }

    return data_pack - pack_size;
}

void depacking_col_inplace_int8(int8_t *data_pack, size_t M, size_t N, size_t m)
{
    assert(0 == M % m);
    size_t i, j, k;
    size_t pack_size = m * N;

    int8_t *data = (int8_t *)malloc(sizeof(int8_t) * M * N);
    memset(data, 0, sizeof(int8_t) * M * N);

    for (i = 0; i < M / m; ++i) {
        for (j = 0; j < m; ++j) {
            for (k = 0; k < N; ++k) {
                *data++ = *data_pack;
                data_pack += m;
            }
            data_pack -= pack_size - 1;
        }
        data_pack += pack_size - m;
    }

    data -= M * N;
    data_pack -= M * N;
    memcpy(data_pack, data, sizeof(int8_t) * M * N);
    free(data);
}

void depacking_col_inplace_int32(int32_t *data_pack, const size_t M, const size_t N, const size_t m)
{
    assert(0 == M % m);
    size_t i, j, k;
    size_t pack_size = m * N;

    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * M * N);
    memset(data, 0, sizeof(int32_t) * M * N);

    for (i = 0; i < M / m; ++i) {
        for (j = 0; j < m; ++j) {
            for (k = 0; k < N; ++k) {
                *data++ = *data_pack;
                data_pack += m;
            }
            data_pack -= pack_size - 1;
        }
        data_pack += pack_size - m;
    }

    data -= M * N;
    data_pack -= M * N;
    memcpy(data_pack, data, sizeof(int32_t) * M * N);
    free(data);
}

#if defined(__aarch64__) && !defined(__APPLE__)
void memcpy_neon64(void *dst, const void *src, size_t count)
{
    // clang-format off
__asm__ __volatile__(
"mov x0, %[src]\n\t"
"mov x1, %[dst]\n\t"
"int8_memcpy_dim64_loop_%=:\n\t"
"cmp %[count], #64\n\t"
"blt int8_finish_memcpy_dim64_loop_%=\n\t"

"ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64\n\t"
"st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64\n\t"

"sub %[count], %[count], #64\n\t"
"b int8_memcpy_dim64_loop_%=\n\t"
"int8_finish_memcpy_dim64_loop_%=:\n\t"
"int8_memcpy_dim48_loop_%=:\n\t"
"cmp %[count], #48\n\t"
"blt int8_finish_memcpy_dim48_loop_%=\n\t"

"ld1 {v0.16b, v1.16b, v2.16b}, [x0], #48\n\t"
"st1 {v0.16b, v1.16b, v2.16b}, [x1], #48\n\t"

"sub %[count], %[count], #48\n\t"
"b int8_memcpy_dim48_loop_%=\n\t"
"int8_finish_memcpy_dim48_loop_%=:\n\t"
"int8_memcpy_dim32_loop_%=:\n\t"
"cmp %[count], #32\n\t"
"blt int8_finish_memcpy_dim32_loop_%=\n\t"

"ld1 {v0.16b, v1.16b}, [x0], #32\n\t"
"st1 {v0.16b, v1.16b}, [x1], #32\n\t"

"sub %[count], %[count], #32\n\t"
"b int8_memcpy_dim32_loop_%=\n\t"
"int8_finish_memcpy_dim32_loop_%=:\n\t"
"int8_memcpy_dim16_loop_%=:\n\t"
"cmp %[count], #16\n\t"
"blt int8_finish_memcpy_dim16_loop_%=\n\t"

"ld1 {v0.16b}, [x0], #16\n\t"
"st1 {v0.16b}, [x1], #16\n\t"

"sub %[count], %[count], #16\n\t"
"b int8_memcpy_dim16_loop_%=\n\t"
"int8_finish_memcpy_dim16_loop_%=:\n\t"
"int8_memcpy_dim8_loop_%=:\n\t"
"cmp %[count], #8\n\t"
"blt int8_finish_memcpy_dim8_loop_%=\n\t"

"ld1 {v0.d}[0], [x0], #8\n\t"
"st1 {v0.d}[0], [x1], #8\n\t"

"sub %[count], %[count], #8\n\t"
"b int8_memcpy_dim8_loop_%=\n\t"
"int8_finish_memcpy_dim8_loop_%=:\n\t"
"int8_memcpy_dim4_loop_%=:\n\t"
"cmp %[count], #4\n\t"
"blt int8_finish_memcpy_dim4_loop_%=\n\t"

"ld1 {v0.s}[0], [x0], #4\n\t"
"st1 {v0.s}[0], [x1], #4\n\t"

"sub %[count], %[count], #4\n\t"
"b int8_memcpy_dim4_loop_%=\n\t"
"int8_finish_memcpy_dim4_loop_%=:\n\t"
"int8_memcpy_dim1_loop_%=:\n\t"
"cmp %[count], #0\n\t"
"beq int8_finish_memcpy_dim1_loop_%=\n\t"

"ld1 {v0.b}[0], [x0], #1\n\t"
"st1 {v0.b}[0], [x1], #1\n\t"

"sub %[count], %[count], #1\n\t"
"b int8_memcpy_dim1_loop_%=\n\t"
"int8_finish_memcpy_dim1_loop_%=:\n\t"
: //output
: //input
[src] "r" (src),
[dst] "r" (dst),
[count] "r" (count)
: //clobbers
"cc", "memory", "x0", "x1",
"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
"v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
"v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
"v25", "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
}
#endif
