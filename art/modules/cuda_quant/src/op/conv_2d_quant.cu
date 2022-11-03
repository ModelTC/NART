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

#include "conv_2d_quant.cuh"

#include "../conv/conv_int4_template_table.cuh"
#include "../conv/conv_int8_template_table.cuh"
#include "../utils/cuda_quant_helper.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#define PROFILE_TEST_TIMES 10

static void pre_compute_matrixA_index(preComputeMatixA *preCompute, op_t *op, int MAX_M_tiling)
{

    shape_t output_shape = op->output_tensors[0].shape;
    shape_t *input_shape = &op->input_tensors[0]->shape;

    // todo
    int out_batch = output_shape.dim[0];
    int out_channel = output_shape.dim[1];
    int out_height = output_shape.dim[2];
    int out_width = output_shape.dim[3];

    int i_c = input_shape->dim[1];
    int i_h = input_shape->dim[2];
    int i_w = input_shape->dim[3];

    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;

    int str_h = conv_op->stride_h;
    int str_w = conv_op->stride_w;
    int pad_h = conv_op->pad_h;
    int pad_w = conv_op->pad_w;

    int matirx_A_row = out_batch * out_height * out_width;
    //    int M_tiling = int8_gemm_table_tiling[best_tiling_index][0];
    int M_tile_num = (matirx_A_row + MAX_M_tiling - 1) / MAX_M_tiling;

    int len = MAX_M_tiling * M_tile_num;
    preComputeMatixA *preCompute_hptr = (preComputeMatixA *)malloc(sizeof(preComputeMatixA) * len);

    for (int j = 0; j < M_tile_num; j++) {
        for (int i = 0; i < MAX_M_tiling; i++) {
            int row_index = (j * MAX_M_tiling + i);
            int batch = row_index / (out_height * out_width);
            int out_height_idx = (row_index / out_width) % out_height;
            int out_width_idx = row_index % out_width;

            int im_h = out_height_idx * str_h - pad_h;
            int im_w = out_width_idx * str_w - pad_w;
            uint8_t out_filled_zero = row_index >= out_batch * out_height * out_width ? 1 : 0;

            int32_t image_offset = batch * i_c * i_h * i_w + im_h * i_w * i_c + im_w * i_c;
            preComputeMatixA t;
            t.image_offset = image_offset;
            t.out_height = out_height_idx;
            t.out_width = out_width_idx;
            t.im_h = im_h;
            t.im_w = im_w;
            t.batch = batch;
            t.fill_zero_flag = out_filled_zero;
            t.aligned[0] = 0;
            t.aligned[1] = 0;
            preCompute_hptr[j * MAX_M_tiling + i] = t;
        }
    }
    cudaMemcpy(preCompute, preCompute_hptr, sizeof(preComputeMatixA) * len, cudaMemcpyHostToDevice);
    free(preCompute_hptr);
    return;
}

static int
profile_int8_best_tiling_size(int *best_M_tiling, int *best_N_tiling, int *best_K_tiling, op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;

    shape_t *output_shape = &op->output_tensors[0].shape;
    shape_t *input_shape = &op->input_tensors[0]->shape;
    shape_t *weight_shape = &op->input_tensors[1]->shape;

    // todo
    int o_b = output_shape->dim[0];
    int o_c = output_shape->dim[1];
    int o_h = output_shape->dim[2];
    int o_w = output_shape->dim[3];

    int i_b = input_shape->dim[0];
    int i_c = input_shape->dim[1];
    int i_h = input_shape->dim[2];
    int i_w = input_shape->dim[3];

    int k_n = o_c;
    int k_c = i_c;
    int k_h = weight_shape->dim[2];
    int k_w = weight_shape->dim[3];

    // All in NHWC pattern
    int M_raw = o_b * o_h * o_w;
    int N_raw = o_c;
    int K_raw = k_c * k_h * k_w;

    uint64_t FLOPS = (uint64_t)M_raw * N_raw * K_raw;

    int out_bit = conv_op->obits[0];

    int input_sz = shape_count(input_shape);
    int weight_sz = shape_count(weight_shape);
    int output_sz = shape_count(output_shape);

    int preMatrixAlen = (M_raw + 255) / 256 * 256;
    char *mat_a, *mat_b, *mat_c;
    mat_a = (char *)mem_data(op->input_tensors[0]->mem);
    mat_b = (char *)mem_data(op->input_tensors[1]->mem);
    mat_c = (char *)mem_data(op->output_tensors->mem);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float max_tflops = 0.0f, max_split_tflops = 0.0f;
    // float min_tflops = 100.f;
    int max_func_idx = -1; // min_func_idx = -1;
    std::function<void(
        char *, char *, char *, int, int, int, int, int, int, int, int, int, int, int, int, float *,
        bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_best_func = NULL;
    std::function<void(
        char *, char *, char *, int *, int, int, int, int, int, int, int, int, int, int, int, int,
        float *, bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_split_best_func = NULL;

    preComputeMatixA *pre_matrixA_ptr;
    cudaMalloc((void **)&pre_matrixA_ptr, sizeof(preComputeMatixA) * preMatrixAlen);
    pre_compute_matrixA_index(pre_matrixA_ptr, op, 256);

    int *tmp = (int *)mem_data(conv_op->ws_inttmp);

    int best_K_tiling_s, best_M_tiling_s, best_N_tiling_s;

    CHECK(i_c >= 32 && (i_c % 32) == 0);
    // todo
    //  if(i_c >= 256 && (i_c % 256) == 0){
    if (i_c >= 256) {
        for (unsigned int i = 0; i < int8_conv_K256_table_tiling.size(); i++) {
            float tflops = 0.0f;
            ;
            if (o_c >= int8_conv_K256_table_tiling[i][1]) {
                float time;
                cudaEventRecord(start);
                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int8_conv_K256_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);
                // printf("128 :%s \n", cudaGetErrorString(cudaGetLastError()));
                CUDA_CHECK(cudaPeekAtLastError());
                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }

            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int8_conv_K256_table[i];
                *best_M_tiling = int8_conv_K256_table_tiling[i][0];
                *best_N_tiling = int8_conv_K256_table_tiling[i][1];
                *best_K_tiling = int8_conv_K256_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int8_conv_K256_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                ;
                // if(o_c >= int8_conv_K256_split3x3_table_tiling[i][1])
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int8_conv_K256_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, NULL, false, false,
                            out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int8_conv_K256_split3x3_table[i];
                    best_M_tiling_s = int8_conv_K256_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int8_conv_K256_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int8_conv_K256_split3x3_table_tiling[i][2];
                }
            }
        }
    }

    // if(i_c >= 128 && (i_c % 128) == 0){
    if (i_c >= 128) {
        for (unsigned int i = 0; i < int8_conv_K128_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int8_conv_K128_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);
                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int8_conv_K128_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);
                CUDA_CHECK(cudaPeekAtLastError());
                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }

            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int8_conv_K128_table[i];
                *best_M_tiling = int8_conv_K128_table_tiling[i][0];
                *best_N_tiling = int8_conv_K128_table_tiling[i][1];
                *best_K_tiling = int8_conv_K128_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int8_conv_K128_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                // if(o_c >= int8_conv_K128_split3x3_table_tiling[i][1])
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int8_conv_K128_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, NULL, false, false,
                            out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int8_conv_K128_split3x3_table[i];
                    best_M_tiling_s = int8_conv_K128_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int8_conv_K128_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int8_conv_K128_split3x3_table_tiling[i][2];
                }
            }
        }
    }
    // if(i_c >= 64 && (i_c % 64) == 0){
    if (i_c >= 64) {
        for (unsigned int i = 0; i < int8_conv_K64_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int8_conv_K64_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);

                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int8_conv_K64_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }
            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int8_conv_K64_table[i];
                *best_M_tiling = int8_conv_K64_table_tiling[i][0];
                *best_N_tiling = int8_conv_K64_table_tiling[i][1];
                *best_K_tiling = int8_conv_K64_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int8_conv_K64_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                // if(o_c >= int8_conv_K64_split3x3_table_tiling[i][1]){
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int8_conv_K64_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, (float *)NULL, false,
                            false, out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int8_conv_K64_split3x3_table[i];
                    best_M_tiling_s = int8_conv_K64_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int8_conv_K64_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int8_conv_K64_split3x3_table_tiling[i][2];
                }
            }
        }
    }

    // if(i_c >= 32 && (i_c % 32) == 0 )
    {
        for (unsigned int i = 0; i < int8_conv_K32_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int8_conv_K32_table_tiling[i][1]){
            {
                float time;
                cudaEventRecord(start);

                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int8_conv_K32_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }
            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int8_conv_K32_table[i];
                *best_M_tiling = int8_conv_K32_table_tiling[i][0];
                *best_N_tiling = int8_conv_K32_table_tiling[i][1];
                *best_K_tiling = int8_conv_K32_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int8_conv_K32_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                // if(o_c >= int8_conv_K32_split3x3_table_tiling[i][1]){
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int8_conv_K32_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, (float *)NULL, false,
                            false, out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int8_conv_K32_split3x3_table[i];
                    best_M_tiling_s = int8_conv_K32_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int8_conv_K32_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int8_conv_K32_split3x3_table_tiling[i][2];
                }
            }
        }
    }

    if (max_tflops < max_split_tflops) {
        conv_op->use_conv_split = true;
        conv_op->conv_best_func_split = conv_split_best_func;
        *best_M_tiling = best_M_tiling_s;
        *best_N_tiling = best_N_tiling_s;
        *best_K_tiling = best_K_tiling_s;
    }
    conv_op->conv_best_func = conv_best_func;

    cudaFree(pre_matrixA_ptr);

    return max_func_idx;
}

static int
profile_int4_best_tiling_size(int *best_M_tiling, int *best_N_tiling, int *best_K_tiling, op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    shape_t *output_shape = &op->output_tensors[0].shape;
    shape_t *input_shape = &op->input_tensors[0]->shape;
    shape_t *weight_shape = &op->input_tensors[1]->shape;

    // todo
    int o_b = output_shape->dim[0];
    int o_c = output_shape->dim[1];
    int o_h = output_shape->dim[2];
    int o_w = output_shape->dim[3];

    int i_b = input_shape->dim[0];
    int i_c = input_shape->dim[1];
    int i_h = input_shape->dim[2];
    int i_w = input_shape->dim[3];

    int k_n = o_c;
    int k_c = i_c;
    int k_h = weight_shape->dim[2];
    int k_w = weight_shape->dim[3];

    // All in NHWC pattern
    int M_raw = o_b * o_h * o_w;
    int N_raw = o_c;
    int K_raw = k_c * k_h * k_w;

    uint64_t FLOPS = (uint64_t)M_raw * N_raw * K_raw;

    int out_bit = conv_op->obits[0];
    // todo!!!!
    out_bit = 4;
    int preMatrixAlen = (M_raw + 255) / 256 * 256; // The maximum padding of matrix A is 256

    char *mat_a, *mat_b, *mat_c;
    mat_a = (char *)mem_data(op->input_tensors[0]->mem);
    mat_b = (char *)mem_data(op->input_tensors[1]->mem);
    mat_c = (char *)mem_data(op->output_tensors->mem);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float max_tflops = 0.0f, max_split_tflops = 0.0f;
    // float min_tflops = 100.f;
    int max_func_idx = -1; // min_func_idx = -1;
    std::function<void(
        char *, char *, char *, int, int, int, int, int, int, int, int, int, int, int, int, float *,
        bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_best_func = NULL;
    std::function<void(
        char *, char *, char *, int *, int, int, int, int, int, int, int, int, int, int, int, int,
        float *, bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_split_best_func = NULL;

    preComputeMatixA *pre_matrixA_ptr;
    cudaMalloc((void **)&pre_matrixA_ptr, sizeof(preComputeMatixA) * preMatrixAlen);
    pre_compute_matrixA_index(pre_matrixA_ptr, op, 256);

    int *tmp = (int *)mem_data(conv_op->ws_inttmp);

    int best_K_tiling_s, best_M_tiling_s, best_N_tiling_s;

    CHECK(i_c >= 32 && (i_c % 32) == 0);

    // if(i_c >= 256 && (i_c % 256) == 0){
    if (i_c >= 256) {
        for (unsigned int i = 0; i < int4_conv_K256_table_tiling.size(); i++) {
            float tflops = 0.0f;
            ;
            // if(o_c >= int4_conv_K256_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);
                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int4_conv_K256_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);
                // printf("128 :%s \n", cudaGetErrorString(cudaGetLastError()));
                CUDA_CHECK(cudaPeekAtLastError());
                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }

            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int4_conv_K256_table[i];
                *best_M_tiling = int4_conv_K256_table_tiling[i][0];
                *best_N_tiling = int4_conv_K256_table_tiling[i][1];
                *best_K_tiling = int4_conv_K256_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int4_conv_K256_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                ;
                // if(o_c >= int4_conv_K256_split3x3_table_tiling[i][1])
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int4_conv_K256_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, NULL, false, false,
                            out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int4_conv_K256_split3x3_table[i];
                    best_M_tiling_s = int4_conv_K256_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int4_conv_K256_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int4_conv_K256_split3x3_table_tiling[i][2];
                }
            }
        }
    }

    // if(i_c >= 128 && (i_c % 128) == 0){
    if (i_c >= 128) {
        for (unsigned int i = 0; i < int4_conv_K128_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int4_conv_K128_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);
                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int4_conv_K128_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);
                CUDA_CHECK(cudaPeekAtLastError());
                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }

            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int4_conv_K128_table[i];
                *best_M_tiling = int4_conv_K128_table_tiling[i][0];
                *best_N_tiling = int4_conv_K128_table_tiling[i][1];
                *best_K_tiling = int4_conv_K128_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int4_conv_K128_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                // if(o_c >= int4_conv_K128_split3x3_table_tiling[i][1])
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int4_conv_K128_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, NULL, false, false,
                            out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int4_conv_K128_split3x3_table[i];
                    best_M_tiling_s = int4_conv_K128_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int4_conv_K128_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int4_conv_K128_split3x3_table_tiling[i][2];
                }
            }
        }
    }
    // if(i_c >= 64 && (i_c % 64) == 0){
    if (i_c >= 64) {
        for (unsigned int i = 0; i < int4_conv_K64_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int4_conv_K64_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);

                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int4_conv_K64_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }
            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int4_conv_K64_table[i];
                *best_M_tiling = int4_conv_K64_table_tiling[i][0];
                *best_N_tiling = int4_conv_K64_table_tiling[i][1];
                *best_K_tiling = int4_conv_K64_table_tiling[i][2];
            }
        }
        if (k_h > 1 && o_b == 1) {
            for (unsigned int i = 0; i < int4_conv_K64_split3x3_table_tiling.size(); i++) {
                float tflops = 0.0f;
                // if(o_c >= int4_conv_K64_split3x3_table_tiling[i][1])
                {
                    float time;
                    cudaEventRecord(start);
                    for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                        int4_conv_K64_split3x3_table[i](
                            (char *)mat_a, (char *)mat_b, (char *)mat_c, tmp, i_b, i_c, i_h, i_w,
                            k_n, k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w,
                            (int)conv_op->stride_h, (int)conv_op->stride_w, NULL, false, false,
                            out_bit, (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr,
                            op->workspace);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(start);
                    cudaEventSynchronize(stop);

                    CUDA_CHECK(cudaPeekAtLastError());
                    cudaEventElapsedTime(&time, start, stop);
                    tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
                }

                if (tflops > max_split_tflops) {
                    max_split_tflops = tflops;
                    conv_split_best_func = int4_conv_K64_split3x3_table[i];
                    best_M_tiling_s = int4_conv_K64_split3x3_table_tiling[i][0];
                    best_N_tiling_s = int4_conv_K64_split3x3_table_tiling[i][1];
                    best_K_tiling_s = int4_conv_K64_split3x3_table_tiling[i][2];
                }
            }
        }
    }

    {
        for (unsigned int i = 0; i < int4_conv_K32_table_tiling.size(); i++) {
            float tflops = 0.0f;
            // if(o_c >= int4_conv_K32_table_tiling[i][1])
            {
                float time;
                cudaEventRecord(start);

                for (int t = 0; t < PROFILE_TEST_TIMES; t++) {
                    int4_conv_K32_table[i](
                        (char *)mat_a, (char *)mat_b, (char *)mat_c, i_b, i_c, i_h, i_w, k_n, k_c,
                        k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
                        (int)conv_op->stride_w, NULL, false, false, out_bit,
                        (float *)mem_data(conv_op->ws_alphas), pre_matrixA_ptr, op->workspace);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(start);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&time, start, stop);
                tflops = FLOPS * PROFILE_TEST_TIMES / (time) / 1000000000.f * 2;
            }
            if (tflops > max_tflops) {
                max_tflops = tflops;
                conv_best_func = int4_conv_K32_table[i];
                *best_M_tiling = int4_conv_K32_table_tiling[i][0];
                *best_N_tiling = int4_conv_K32_table_tiling[i][1];
                *best_K_tiling = int4_conv_K32_table_tiling[i][2];
            }
        }
    }
    // printf("max split_tflops: %.3f, orignal tflops:%.3f\n", max_split_tflops, max_tflops);
    if (max_tflops < max_split_tflops) {
        conv_op->use_conv_split = true;
        conv_op->conv_best_func_split = conv_split_best_func;
        *best_M_tiling = best_M_tiling_s;
        *best_N_tiling = best_N_tiling_s;
        *best_K_tiling = best_K_tiling_s;
    }
    conv_op->conv_best_func = conv_best_func;

    cudaFree(pre_matrixA_ptr);

    return max_func_idx;
}

static void calculate_perchannel_alphas(
    mem_t **ws_alphas, int out_channel, float *ialphas, float *walphas, float *oalphas)
{

    mem_alloc(*ws_alphas, sizeof(float) * out_channel);
    float *alphas_host = (float *)malloc(sizeof(float) * out_channel);
    for (int i = 0; i < out_channel; i++) {
        // todo
        alphas_host[i] = ialphas[0] * walphas[i] / oalphas[0];
    }
    cudaMemcpy(
        mem_data(*ws_alphas), alphas_host, sizeof(float) * out_channel, cudaMemcpyHostToDevice);
    free(alphas_host);
}

op_conv_2d_t *op_cuda_quant_conv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_conv_2d_t *res = new op_conv_2d_t;
    memset(res, 0, sizeof(op_conv_2d_t));

    res->ws_pre = mem_new(cuda_mem_tp);
    res->ws_alphas = mem_new(cuda_mem_tp);
    res->ws_inttmp = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_quant_conv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_conv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_conv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_conv_2d_t *)op)->kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_conv_2d_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_conv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_conv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_conv_2d_t *)op)->stride_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_conv_2d_t *)op)->group));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_RELU_FLAG, dtBOOL, &((op_conv_2d_t *)op)->relu_flag));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_WZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->wzero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_WBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_conv_2d_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_conv_2d_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OBITS, dtUINT8, &len_bits, &((op_conv_2d_t *)op)->obits));
}

void op_cuda_quant_conv_2d_tp_destroy(op_t *op) { (void)op; }

void op_cuda_quant_conv_2d_tp_dealloc(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    if (NULL != conv_op->ws_pre) {
        mem_delete(conv_op->ws_pre);
    }
    if (NULL != conv_op->ws_alphas) {
        mem_delete(conv_op->ws_alphas);
    }
    if (NULL != conv_op->ws_inttmp) {
        mem_delete(conv_op->ws_inttmp);
    }
    delete conv_op;
}

void op_cuda_quant_conv_2d_tp_prepare(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;

    int in_bit = conv_op->wbits[0];

    /* set input tensor desc */
    shape_t *input_shape = &op->input_tensors[0]->shape;
    shape_t *kernel_shape = &op->input_tensors[1]->shape;
    shape_t *output_shape = &op->output_tensors[0].shape;

    // todo
    int i_c = input_shape->dim[1];

    // todo
    //  in NHWC pattern
    size_t o_b = output_shape->dim[0];
    size_t o_c = output_shape->dim[1];
    size_t o_h = output_shape->dim[2];
    size_t o_w = output_shape->dim[3];
    // todo!

    // printf("%d %d %d %d %d %d %d %d %.10f %.10f %.10f %d\n", input_shape->dim[0],
    // input_shape->dim[2], input_shape->dim[3], 	input_shape->dim[1], conv_op->num_output,
    // conv_op->kernel_h, conv_op->pad_h, conv_op->stride_h, 	conv_op->ialpha[0],
    // conv_op->walpha[0], conv_op->oalpha[0], conv_op->relu_flag ? 1 : 0);

    // in_bit = 4;
    // if(i_c == 32)
    //    in_bit = 8;
    int d_tp = op->input_tensors[0]->dtype;
    mem_alloc(conv_op->ws_inttmp, sizeof(int) * shape_count(output_shape));
    calculate_perchannel_alphas(
        &conv_op->ws_alphas, o_c, conv_op->ialpha, conv_op->walpha, conv_op->oalpha);

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    switch (in_bit) {
    case 8: {
        int best_M_tiling, best_N_tiling, best_K_tiling;
        int best_func_index
            = profile_int8_best_tiling_size(&best_M_tiling, &best_N_tiling, &best_K_tiling, op);

        conv_op->M_best_pad = best_M_tiling;
        conv_op->N_best_pad = best_N_tiling;
        conv_op->K_best_pad = best_K_tiling;

        int M_tiling = conv_op->M_best_pad;
        int M_tile_num = (o_b * o_h * o_w + M_tiling - 1) / M_tiling;

        int len = M_tiling * M_tile_num;

        mem_alloc(conv_op->ws_pre, sizeof(preComputeMatixA) * len);

        preComputeMatixA *pre_ptr = (preComputeMatixA *)mem_data(conv_op->ws_pre);

        pre_compute_matrixA_index(pre_ptr, op, best_M_tiling);

        if (d_tp == dtINT8)
            op->run_func = op_cuda_quant_conv_2d_i8xi8_run;
        else
            CHECK(false);
        break;
    }
    case 4: {
        int best_M_tiling, best_N_tiling, best_K_tiling;
        int best_func_index
            = profile_int4_best_tiling_size(&best_M_tiling, &best_N_tiling, &best_K_tiling, op);

        conv_op->M_best_pad = best_M_tiling;
        conv_op->N_best_pad = best_N_tiling;
        conv_op->K_best_pad = best_K_tiling;

        int M_tiling = conv_op->M_best_pad;
        int M_tile_num = (o_b * o_h * o_w + M_tiling - 1) / M_tiling;

        int len = M_tiling * M_tile_num;

        mem_alloc(conv_op->ws_pre, sizeof(preComputeMatixA) * len);
        preComputeMatixA *pre_ptr = (preComputeMatixA *)mem_data(conv_op->ws_pre);

        pre_compute_matrixA_index(pre_ptr, op, best_M_tiling);

        if (d_tp == dtINT8)
            op->run_func = op_cuda_quant_conv_2d_i4xi4_run;
        else
            CHECK(false);
        break;
    }
    default: {
        CHECK(false);
    }
    }
}

#ifdef __cplusplus
}
#endif
