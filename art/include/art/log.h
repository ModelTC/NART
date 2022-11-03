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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define FILE_POS(file, pos)             file ":" #pos
#define FORM_LOG_HEAD(type, file, line) "[" #type "]:" FILE_POS(file, line) "\n"

#define DO_NOTHING \
    do {           \
    } while (0)

#ifndef NDEBUG
#define LOG_error(...)                                             \
    do {                                                           \
        fprintf(stderr, "[Error]: %s:%d\n\t", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                              \
        exit(-1);                                                  \
    } while (0)

#define LOG_info(...)                                           \
    do {                                                        \
        fprintf(stderr, "[Info]: %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                           \
    } while (0)

#define LOG_warn(...)                                              \
    do {                                                           \
        fprintf(stderr, "[Warning]: %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                              \
    } while (0)

#else
#define LOG_error(...)                 \
    do {                               \
        fprintf(stderr, "[Error]:\t"); \
        fprintf(stderr, __VA_ARGS__);  \
        exit(-1);                      \
    } while (0)

#define LOG_info(...)                 \
    do {                              \
        fprintf(stderr, "[Info]:\t"); \
        fprintf(stderr, __VA_ARGS__); \
    } while (0)

#define LOG_warn(...)                    \
    do {                                 \
        fprintf(stderr, "[Warning]:\t"); \
        fprintf(stderr, __VA_ARGS__);    \
    } while (0)

#endif

#if NART_LOG_LEVEL <= 0
#define LOG_debug(...)                                            \
    do {                                                          \
        fprintf(stderr, "[Debug]@ %s:%d : ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                             \
    } while (0)
#else
#define LOG_debug(...) DO_NOTHING
#endif

#define LOG(level, ...) LOG_##level(__VA_ARGS__)

#define CHECK_ACTION(p)                                                                           \
    if (true != (p)                                                                               \
        && (fputs(FORM_LOG_HEAD(Error, __FILE__, __LINE__) "\tCheck failed: (" #p ")\n", stderr), \
            1))

#define CHECK(p)                    \
    CHECK_ACTION(p) do { abort(); } \
    while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

#define CHECK_NE(a, b) CHECK((a) != (b))

#define CHECK_GT(a, b) CHECK((a) > (b))

#define CHECK_GE(a, b) CHECK((a) >= (b))

#define CHECK_LT(a, b) CHECK((a) < (b))

#define CHECK_LE(a, b) CHECK((a) <= (b))

#ifndef NDEBUG
#define DCHECK    CHECK
#define DCHECK_EQ CHECK_EQ
#define DCHECK_NE CHECK_NE
#define DCHECK_GT CHECK_GT
#define DCHECK_GE CHECK_GE
#define DCHECK_LT CHECK_LT
#define DCHECK_LE CHECK_LE
#define DLOG      LOG
#else
#define DCHECK(...)
#define DCHECK_EQ(...)
#define DCHECK_NE(...)
#define DCHECK_GT(...)
#define DCHECK_GE(...)
#define DCHECK_LT(...)
#define DCHECK_LE(...)
#define DLOG(...)
#endif

#define CAT(a, b) _CAT(a, b)

#define _CAT(a, b) a##b

#define CUDA_CHECK_ACTION(p)                                                                      \
    if (cudaSuccess != (p)                                                                        \
        && (fputs(                                                                                \
                FORM_LOG_HEAD(Error, __FILE__, __LINE__) "\tCheck failed: (" #p ")\n\t", stderr), \
            fprintf(stderr, "message: %s\n", cudaGetErrorString(cudaGetLastError())), 1))

#define CUDA_CHECK(p)                    \
    CUDA_CHECK_ACTION(p) do { abort(); } \
    while (0)

#define CUDNN_CHECK_ACTION(p)                                                                     \
    cudnnStatus_t CAT(_cudnn_res_, __LINE__) = p;                                                 \
    if (CUDNN_STATUS_SUCCESS != CAT(_cudnn_res_, __LINE__)                                        \
        && (fputs(                                                                                \
                FORM_LOG_HEAD(Error, __FILE__, __LINE__) "\tCheck failed: (" #p ")\n\t", stderr), \
            fprintf(stderr, "message: %s\n", cudnnGetErrorString(CAT(_cudnn_res_, __LINE__))), 1))

#define CUDNN_CHECK(p)                    \
    CUDNN_CHECK_ACTION(p) do { abort(); } \
    while (0)

#define CUBLAS_CHECK_ACTION(p)                                                                    \
    cublasStatus_t CAT(_cublas_res_, __LINE__) = p;                                               \
    if (CUBLAS_STATUS_SUCCESS != CAT(_cublas_res_, __LINE__)                                      \
        && (fputs(                                                                                \
                FORM_LOG_HEAD(Error, __FILE__, __LINE__) "\tCheck failed: (" #p ")\n\t", stderr), \
            fprintf(stderr, "message: %s\n", cublasGetErrorString(CAT(_cublas_res_, __LINE__))),  \
            1))

#define CUBLAS_CHECK(p)                    \
    CUBLAS_CHECK_ACTION(p) do { abort(); } \
    while (0)

#define MKLDNN_CHECK(f)                                                          \
    do {                                                                         \
        mkldnn_status_t s = f;                                                   \
        if (s != mkldnn_success) {                                               \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
            abort();                                                             \
        }                                                                        \
    } while (0)
