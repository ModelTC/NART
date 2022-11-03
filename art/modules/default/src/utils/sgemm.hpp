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

#ifndef SGEMM_HPP
#define SGEMM_HPP

#include <string.h>

#include "art/mem.h"

template <typename T1, typename T2, typename T3>
void sgemm(
    const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C,
    mem_t *aux_mem_1, mem_t *aux_mem_2);
template <typename T1, typename T2, typename T3>
void sgemm_AxB(const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C);
template <typename T1, typename T2, typename T3>
void sgemm_ATxB(
    const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C,
    mem_t *aux_mem_1, mem_t *aux_mem_2);

// 调用完成后注意释放临时申请的空间
template <typename T> void transpose(const T *data, mem_t *aux_mem, size_t M, size_t N)
{
    size_t i, j;
    size_t count = M * N;
    T *data_trans = (T *)mem_data(aux_mem);

    for (i = N; i > 0; --i) {
        for (j = M; j > 0; --j) {
            *data_trans++ = *data;
            data += N;
        }
        data -= count - 1;
    }
}

template <typename T> void transpose_inplace(T *data, mem_t *aux_mem, size_t M, size_t N)
{
    size_t i, j;
    size_t count = M * N;
    T *data_trans = (T *)mem_data(aux_mem);

    for (i = N; i > 0; --i) {
        for (j = M; j > 0; --j) {
            *data_trans++ = *data;
            data += N;
        }
        data -= count - 1;
    }
    data -= N;
    data_trans -= count;

    memcpy(data, data_trans, sizeof(T) * count);
}

template <typename T1, typename T2, typename T3>
void sgemm(
    const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C,
    mem_t *aux_mem_1, mem_t *aux_mem_2)
{
    size_t i, j, k;
    const T1 *A_temp;
    const T2 *B_temp;
    T3 *C_temp;

    // TODO parameter check

    A_temp = A;
    B_temp = B;
    C_temp = C;

    size_t count_a = M * K;
    transpose(A, aux_mem_1, M, K);
    A_temp = (const T1 *)mem_data(aux_mem_1);
    transpose_inplace(C, aux_mem_2, M, N);
    for (i = N; i > 0; --i) {
        for (j = K; j > 0; --j) {
            T2 b_temp = B_temp[0];
            if (0 == b_temp) {
                A_temp += M;
            } else {
                for (k = M; k >= 8; k -= 8) {
                    C_temp[0] += b_temp * A_temp[0];
                    C_temp[1] += b_temp * A_temp[1];
                    C_temp[2] += b_temp * A_temp[2];
                    C_temp[3] += b_temp * A_temp[3];
                    C_temp[4] += b_temp * A_temp[4];
                    C_temp[5] += b_temp * A_temp[5];
                    C_temp[6] += b_temp * A_temp[6];
                    C_temp[7] += b_temp * A_temp[7];
                    A_temp += 8;
                    C_temp += 8;
                }
                while (k-- > 0) {
                    *C_temp++ += b_temp * *A_temp++;
                }
                C_temp -= M;
            }
            B_temp++;
        }
        A_temp -= count_a;
        C_temp += M;
    }
    transpose_inplace(C, aux_mem_2, N, M);
}

template <typename T1, typename T2, typename T3>
void sgemm_AxB(const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C)
{
    size_t i, j, k;
    size_t count_b = N * K;

    for (i = M; i > 0; --i) {
        for (j = N; j > 0; --j) {
            T3 temp[8] = { 0 };
            for (k = K; k >= 8; k -= 8) {
                temp[0] += A[0] * B[0];
                temp[1] += A[1] * B[1];
                temp[2] += A[2] * B[2];
                temp[3] += A[3] * B[3];
                temp[4] += A[4] * B[4];
                temp[5] += A[5] * B[5];
                temp[6] += A[6] * B[6];
                temp[7] += A[7] * B[7];
                A += 8;
                B += 8;
            }
            *C += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            while (k-- > 0) {
                *C += *A++ * *B++;
            }
            ++C;
            A -= K;
        }
        A += K;
        B -= count_b;
    }
}

// can not preload bias
template <typename T1, typename T2, typename T3>
void sgemm_ATxB(
    const size_t M, const size_t N, const size_t K, const T1 *A, const T2 *B, T3 *C,
    mem_t *aux_mem_1, mem_t *aux_mem_2)
{
    int i, j, k;
    size_t count_b = N * K;
    const T1 *A_temp;
    const T2 *B_temp;

    // TODO parameter check

    transpose(A, aux_mem_1, K, M);
    transpose(B, aux_mem_2, K, N);
    A_temp = (const T1 *)mem_data(aux_mem_1);
    B_temp = (const T2 *)mem_data(aux_mem_2);

    for (i = M; i > 0; --i) {
        for (j = N; j > 0; --j) {
            T3 temp[8] = { 0 };
            for (k = K; k >= 8; k -= 8) {
                temp[0] += A_temp[0] * B_temp[0];
                temp[1] += A_temp[1] * B_temp[1];
                temp[2] += A_temp[2] * B_temp[2];
                temp[3] += A_temp[3] * B_temp[3];
                temp[4] += A_temp[4] * B_temp[4];
                temp[5] += A_temp[5] * B_temp[5];
                temp[6] += A_temp[6] * B_temp[6];
                temp[7] += A_temp[7] * B_temp[7];
                A_temp += 8;
                B_temp += 8;
            }
            *C = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            while (k-- > 0) {
                *C += *A_temp++ * *B_temp++;
            }
            ++C;
            A_temp -= K;
        }
        A_temp += K;
        B_temp -= count_b;
    }
}

#endif
