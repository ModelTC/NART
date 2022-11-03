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

#ifndef TURING_INDIRECT_CONV_WMMA_LDG16_K64_CUH
#define TURING_INDIRECT_CONV_WMMA_LDG16_K64_CUH

template <
    int WARP_ROW_WMMA_SZ_CHANGE, int WARP_COL_WMMA_SZ_CHANGE, int WMMA_K_STEP_CHANGE,
    int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K, int BLOCK_ROW_WARP_NUM_CHANGE,
    int BLOCK_COL_WARP_NUM_CHANGE, bool overlap = true>
__global__ void lb_turing_indirect_conv_wmma_int8_ldg16_k64_relu(
    char *input_tensor, char *weight_tensor, char *output_tensor, int GEMM_M, int GEMM_N,
    int GEMM_K, int GEMM_M_PAD, int GEMM_N_PAD, int GEMM_K_PAD, float *bias, bool bias_flag,
    bool relu_flag, int i_b, int i_c, int i_h, int i_w, int k_n, int k_c, int k_h, int k_w,
    int pad_h, int pad_w, int str_h, int str_w, int out_batch, int out_channel, int out_height,
    int out_width, uint8_t out_bits, float *alphas, preComputeMatixA *preComputeA)

{

    // one
    const int WMMA_ROW_TOTAL_SZ = WARP_ROW_WMMA_SZ_CHANGE * BLOCK_ROW_WARP_NUM_CHANGE;
    const int WMMA_COL_TOTAL_SZ = WARP_COL_WMMA_SZ_CHANGE * BLOCK_COL_WARP_NUM_CHANGE;

    // shared memory pattern:
    // matrix A: [WMMA_ROW_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_M * WMMA_16x16_K]
    // matrix B: [WMMA_COL_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_N * WMMA_16x16_K]
    const int MATRIX_A_SHMEM_STRIDE = 64; // WMMA_16x16_M * WMMA_16x16_K;//WMMA_K_STEP_CHANGE *
                                          // WMMA_16x16_K + SHMEM_CONFILCT_AVOID;
    const int MATRIX_B_SHMEM_STRIDE = 64; // WMMA_16x16_N * WMMA_16x16_K;

    // Matrix B base address in shared memory
    const int Matrix_B_shmem_offset = WMMA_ROW_TOTAL_SZ * WMMA_16x16_M * MATRIX_A_SHMEM_STRIDE;

    // total warps in a block
    const int WARP_PER_BLOCK_CHANGE = (BLOCK_ROW_WARP_NUM_CHANGE * BLOCK_COL_WARP_NUM_CHANGE);

    // K elements will be processed once at K dimension
    const int K_step = WMMA_K_STEP_CHANGE * WMMA_16x16_K;

    uint32_t warpId = threadIdx.z * blockDim.y + threadIdx.y;
    uint32_t warpIdx = threadIdx.y;
    uint32_t warpIdy = threadIdx.z;

    volatile uint32_t laneId = threadIdx.x;

    uint32_t laneId_even = threadIdx.x / 2;
    uint32_t laneId_ldg_cacheline = threadIdx.x & 1;

    uint32_t laneId_0_8 = threadIdx.x & 0x07;
    // // if a thread is in 0-15 in a warp, will be 0; else will be 1
    uint32_t laneId_quarter = threadIdx.x / 8;

    uint32_t laneId_0_4 = threadIdx.x & 0x03;
    uint32_t laneId_eighth = threadIdx.x / 4;

    //  uint32_t laneId_0_15 = threadIdx.x & 0x0F;
    // // // if a thread is in 0-15 in a warp, will be 0; else will be 1
    //  uint32_t laneId_half = threadIdx.x > 0x0F ? 1 : 0;

    // declare extern shared memory, only will be used if dynamic shared memory used in template
    // function.
    SharedMemory_t<int> shmem;
    const char *shmem_ptr = (char *)shmem.getPointer();
    // extern __shared__ char shmem_ptr[];

    // Matrix A B block index of one Block
    uint32_t pos_x = blockIdx.x;
    uint32_t pos_y = blockIdx.y;

    {
        int2 my_c_frag[WARP_ROW_WMMA_SZ_CHANGE][WARP_COL_WMMA_SZ_CHANGE][4] = { 0 };

        //   | --- block offset        --------------|  |---   warp offset----------------------- |
        int matrix_A_each_warp_actual_row_base = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ
            + warpId * (WMMA_16x16_M * WMMA_ROW_TOTAL_SZ / WARP_PER_BLOCK_CHANGE);
        int matrix_B_each_warp_actual_col_base = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ
            + warpId * (WMMA_16x16_N * WMMA_COL_TOTAL_SZ / WARP_PER_BLOCK_CHANGE);

        int matrx_B_warp_col
            = matrix_B_each_warp_actual_col_base + laneId_eighth; // laneId_even;//+ laneId_0_15;
        int matrx_A_warp_row
            = matrix_A_each_warp_actual_row_base + laneId_eighth; // laneId_even;//+ laneId_0_15;

        preComputeMatixA
            MatrixAInfo[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE * 4)]; // 16
#pragma unroll
        for (int jj = 0; jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
             jj++) { // 16
            preComputeMatixA *MatrixA_info = preComputeA + matrx_A_warp_row + jj * 8;
            MatrixAInfo[jj] = *(preComputeMatixA *)&(__ldg((int4 *)MatrixA_info));
        }

        int i = 0;

        // #pragma unroll
        for (int k_h_index = 0; k_h_index < k_h; k_h_index++) {
            // #pragma unroll
            for (int k_w_index = 0; k_w_index < k_w; k_w_index++) {
                for (int channel_cnt = 0; channel_cnt < i_c; channel_cnt += K_step) {
                    int4 A_row_frag
                        [WMMA_ROW_TOTAL_SZ * WMMA_16x16_M
                         / (8 * WARP_PER_BLOCK_CHANGE)]; //[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M /
                                                         //WARP_PER_BLOCK_CHANGE];
                    int4 B_row_frag
                        [WMMA_COL_TOTAL_SZ * WMMA_16x16_N
                         / (8 * WARP_PER_BLOCK_CHANGE)]; //[WMMA_COL_TOTAL_SZ * WMMA_16x16_N /
                                                         //WARP_PER_BLOCK_CHANGE];

                    // drag matrix weight, 1 warp 1 line(8x16), each thread drag int
#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
                        int4 *src = ((
                            int4
                                *)(weight_tensor + matrx_B_warp_col * GEMM_K_PAD + laneId_0_4 * 16 + jj * 8 * GEMM_K_PAD + (k_h_index * k_w + k_w_index) * i_c + channel_cnt)); // laneId_half * 16
                        B_row_frag[jj] = __ldg(src);
                    }

#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
                        int32_t image_offset = MatrixAInfo[jj].image_offset;
                        int im_h = MatrixAInfo[jj].im_h;
                        int im_w = MatrixAInfo[jj].im_w;
                        bool out_filled_zero = MatrixAInfo[jj].fill_zero_flag == 1 ? true : false;

                        int4 *src = ((
                            int4
                                *)(input_tensor + image_offset + k_h_index * (i_w * i_c) + k_w_index * i_c + channel_cnt + laneId_0_4 * 16)); // laneId_half * 16

                        bool in_window = (im_h + k_h_index >= 0) && (im_h + k_h_index < i_h)
                            && (im_w + k_w_index >= 0) && (im_w + k_w_index < i_w);
                        int4 *src_t = (in_window && (!out_filled_zero)) ? src : zero_arr;
                        A_row_frag[jj]
                            = __ldg(src_t); //(in_window && (!out_filled_zero) && (k_h_index < k_h
                                            //&& k_w_index < k_w)) ? __ldg(src) : zero;
                    }
                    if (!overlap) {
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled
                            int warp_sharedmm_offset = warpId
                                    * (WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_B_SHMEM_STRIDE
                                + jj * 8 * MATRIX_B_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_B_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;

                            int4 *dst = ((int4 *)(&shmem_ptr
                                                      [warp_sharedmm_offset + lane_sharedmm_offset
                                                       + Matrix_B_shmem_offset]));

                            *(dst) = B_row_frag[jj];
                        }

                        // LDG & STS overlap with wmma, now data need to be written to the shared
                        // mm, we assume now the data is ready
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled

                            int warp_sharedmm_offset = warpId
                                    * (WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_A_SHMEM_STRIDE
                                + jj * 8 * MATRIX_A_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_A_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;//

                            int4 *dst = ((
                                int4 *)(&shmem_ptr[warp_sharedmm_offset + lane_sharedmm_offset]));

                            *(dst) = A_row_frag[jj];
                        }
                    }

                    __syncthreads();

                    int cnt = overlap ? 0 : -1;
                    if (i > cnt) {

                        int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
                        int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
                        int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
                        int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];
#pragma unroll
                        for (int k = 0; k < 64; k += 64) {

#pragma unroll
                            for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
                                int warp_offset = (warpIdy * WARP_ROW_WMMA_SZ_CHANGE + row)
                                        * WMMA_16x16_M * MATRIX_A_SHMEM_STRIDE
                                    + k;
                                int lane_offset
                                    = laneId_0_4 * 16 + laneId_eighth * MATRIX_A_SHMEM_STRIDE;
                                int8_t *a_frag0_ptr = (int8_t *)shmem_ptr;
                                int8_t *a_frag1_ptr = a_frag0_ptr + 8 * MATRIX_A_SHMEM_STRIDE;
                                my_A_frag0[row]
                                    = *((int4 *)(&a_frag0_ptr[warp_offset + lane_offset]));
                                my_A_frag1[row]
                                    = *((int4 *)(&a_frag1_ptr[warp_offset + lane_offset]));
                            }

#pragma unroll
                            for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                                int warp_offset = (warpIdx * WARP_COL_WMMA_SZ_CHANGE + col)
                                        * WMMA_16x16_N * MATRIX_B_SHMEM_STRIDE
                                    + k;
                                int lane_offset
                                    = laneId_0_4 * 16 + laneId_eighth * MATRIX_B_SHMEM_STRIDE;
                                int8_t *b_frag0_ptr = (int8_t *)shmem_ptr + Matrix_B_shmem_offset;
                                int8_t *b_frag1_ptr = b_frag0_ptr + 8 * MATRIX_B_SHMEM_STRIDE;
                                my_B_frag0[col]
                                    = *((int4 *)(&b_frag0_ptr[warp_offset + lane_offset]));
                                my_B_frag1[col]
                                    = *((int4 *)(&b_frag1_ptr[warp_offset + lane_offset]));
                            }

#pragma unroll
                            for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
#pragma unroll
                                for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].w,
                                        my_B_frag1[col].w);

                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].w,
                                        my_B_frag1[col].w);
                                }
                            }
                        }
                        __syncthreads();
                    }

                    if (overlap) {
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled
                            int warp_sharedmm_offset = warpId
                                    * (WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_B_SHMEM_STRIDE
                                + jj * 8 * MATRIX_B_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_B_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;

                            int4 *dst = ((int4 *)(&shmem_ptr
                                                      [warp_sharedmm_offset + lane_sharedmm_offset
                                                       + Matrix_B_shmem_offset]));

                            *(dst) = B_row_frag[jj];
                        }

                        // LDG & STS overlap with wmma, now data need to be written to the shared
                        // mm, we assume now the data is ready
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled

                            int warp_sharedmm_offset = warpId
                                    * (WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_A_SHMEM_STRIDE
                                + jj * 8 * MATRIX_A_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_A_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;//

                            int4 *dst = ((
                                int4 *)(&shmem_ptr[warp_sharedmm_offset + lane_sharedmm_offset]));

                            *(dst) = A_row_frag[jj];
                        }
                    }

                    i++;
                }
            }
        }

        if (overlap) {
            int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
            int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
            int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
            int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];

            __syncthreads();
#pragma unroll
            for (int k = 0; k < 64; k += 64) {

#pragma unroll
                for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
                    int warp_offset = (warpIdy * WARP_ROW_WMMA_SZ_CHANGE + row) * WMMA_16x16_M
                            * MATRIX_A_SHMEM_STRIDE
                        + k;
                    int lane_offset = laneId_0_4 * 16 + laneId_eighth * MATRIX_A_SHMEM_STRIDE;
                    int8_t *a_frag0_ptr = (int8_t *)shmem_ptr;
                    int8_t *a_frag1_ptr = a_frag0_ptr + 8 * MATRIX_A_SHMEM_STRIDE;
                    my_A_frag0[row] = *((int4 *)(&a_frag0_ptr[warp_offset + lane_offset]));
                    my_A_frag1[row] = *((int4 *)(&a_frag1_ptr[warp_offset + lane_offset]));
                }

#pragma unroll
                for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                    int warp_offset = (warpIdx * WARP_COL_WMMA_SZ_CHANGE + col) * WMMA_16x16_N
                            * MATRIX_B_SHMEM_STRIDE
                        + k;
                    int lane_offset = laneId_0_4 * 16 + laneId_eighth * MATRIX_B_SHMEM_STRIDE;
                    int8_t *b_frag0_ptr = (int8_t *)shmem_ptr + Matrix_B_shmem_offset;
                    int8_t *b_frag1_ptr = b_frag0_ptr + 8 * MATRIX_B_SHMEM_STRIDE;
                    my_B_frag0[col] = *((int4 *)(&b_frag0_ptr[warp_offset + lane_offset]));
                    my_B_frag1[col] = *((int4 *)(&b_frag1_ptr[warp_offset + lane_offset]));
                }

#pragma unroll
                for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
#pragma unroll
                    for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].x, my_B_frag0[col].x);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].y, my_B_frag0[col].y);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].z, my_B_frag0[col].z);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].w, my_B_frag0[col].w);

                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].x, my_B_frag1[col].x);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].y, my_B_frag1[col].y);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].z, my_B_frag1[col].z);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].w, my_B_frag1[col].w);

                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].x, my_B_frag0[col].x);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].y, my_B_frag0[col].y);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].z, my_B_frag0[col].z);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].w, my_B_frag0[col].w);

                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].x, my_B_frag1[col].x);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].y, my_B_frag1[col].y);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].z, my_B_frag1[col].z);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].w, my_B_frag1[col].w);
                    }
                }
            }
        }
        float2 alpha0[WARP_COL_WMMA_SZ_CHANGE] = { 0 };
        float2 alpha4[WARP_COL_WMMA_SZ_CHANGE] = { 0 };

#pragma unroll
        for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {
            int channel_id_base = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                + j * WMMA_16x16_N // warp dimension offset
                + laneId % 4 * 2;
            int channel_id_frag_reg0 = (channel_id_base + 0);
            int channel_id_frag_reg4 = (channel_id_base + 8);

            alpha0[j] = __ldg((float2 *)(alphas + channel_id_frag_reg0));
            alpha4[j] = __ldg((float2 *)(alphas + channel_id_frag_reg4));
        }

        float2 bias_channel_reg0[WARP_COL_WMMA_SZ_CHANGE] = { 0 };
        float2 bias_channel_reg4[WARP_COL_WMMA_SZ_CHANGE] = { 0 };

        if (bias_flag) {
#pragma unroll
            for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {

                int channel_id_base
                    = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                    + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                    + j * WMMA_16x16_N // warp dimension offset
                    + laneId % 4 * 2;
                int channel_id_frag_reg0
                    = (channel_id_base + 0); // &~(out_channel - 1);//% out_channel;
                int channel_id_frag_reg1
                    = (channel_id_base + 1); // &~(out_channel - 1);// % out_channel;
                                             // int channel_id_frag_reg2 = channel_id_base + 0;
                                             // int channel_id_frag_reg3 = channel_id_base + 1;

                int channel_id_frag_reg4
                    = (channel_id_base + 8); // &~(out_channel - 1);// % out_channel;
                int channel_id_frag_reg5
                    = (channel_id_base + 9); // &~(out_channel - 1);// % out_channel;

                bias_channel_reg0[j] = __ldg((float2 *)(bias + channel_id_frag_reg0));
                bias_channel_reg4[j] = __ldg((float2 *)(bias + channel_id_frag_reg4));
            }
        }

#pragma unroll
        for (int i = 0; i < WARP_ROW_WMMA_SZ_CHANGE; i++) {
#pragma unroll

            for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {

                int channel_id_base
                    = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                    + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                    + j * WMMA_16x16_N // warp dimension offset
                    + laneId % 4 * 2; // thread dimension offset

                int actual_row_base
                    = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ // block dimension offset
                    + warpIdy * WARP_ROW_WMMA_SZ_CHANGE * WMMA_16x16_M // warp dimension offset
                    + i * WMMA_16x16_M // warp dimension offset
                    + laneId / 4; // thread dimension offset

                int actual_col_frag_reg0 = channel_id_base + 0;
                // int actual_col_frag_reg1 = channel_id_base + 1;

                int actual_col_frag_reg4 = channel_id_base + 8;
                // int actual_col_frag_reg5 = channel_id_base + 9;

                // int channel_id_frag_reg0 = (channel_id_base + 0) &~ (out_channel - 1);//%
                // out_channel; int channel_id_frag_reg1 = (channel_id_base + 1) &~ (out_channel -
                // 1);// % out_channel;
                // // int channel_id_frag_reg2 = channel_id_base + 0;
                // // int channel_id_frag_reg3 = channel_id_base + 1;

                // int channel_id_frag_reg4 = (channel_id_base + 8) &~ (out_channel - 1);// %
                // out_channel; int channel_id_frag_reg5 = (channel_id_base + 9) &~ (out_channel -
                // 1);// % out_channel; int channel_id_frag_reg6 = channel_id_base + 8; int
                // channel_id_frag_reg7 = channel_id_base + 9;

                int actual_row_frag_reg0 = actual_row_base + 0;
                int actual_row_frag_reg2 = actual_row_base + 8;

                float res_reg0 = my_c_frag[i][j][0].x + bias_channel_reg0[j].x;
                float res_reg1 = my_c_frag[i][j][0].y + bias_channel_reg0[j].y;
                float res_reg2 = my_c_frag[i][j][1].x + bias_channel_reg4[j].x;
                float res_reg3 = my_c_frag[i][j][1].y + bias_channel_reg4[j].y;

                float res_reg4 = my_c_frag[i][j][2].x + bias_channel_reg0[j].x;
                float res_reg5 = my_c_frag[i][j][2].y + bias_channel_reg0[j].y;
                float res_reg6 = my_c_frag[i][j][3].x + bias_channel_reg4[j].x;
                float res_reg7 = my_c_frag[i][j][3].y + bias_channel_reg4[j].y;

                int32_t quant_res_reg0_alpha = (int)rintf((alpha0[j].x * res_reg0));
                int32_t quant_res_reg1_alpha = (int)rintf((alpha0[j].y * res_reg1));
                int32_t quant_res_reg2_alpha = (int)rintf((alpha4[j].x * res_reg2));
                int32_t quant_res_reg3_alpha = (int)rintf((alpha4[j].y * res_reg3));
                int32_t quant_res_reg4_alpha = (int)rintf((alpha0[j].x * res_reg4));
                int32_t quant_res_reg5_alpha = (int)rintf((alpha0[j].y * res_reg5));
                int32_t quant_res_reg6_alpha = (int)rintf((alpha4[j].x * res_reg6));
                int32_t quant_res_reg7_alpha = (int)rintf((alpha4[j].y * res_reg7));

                // int64_t quant_res_reg0_64 = rshift_rn_int8((int64_t)res_reg0 * (int64_t)res_mult,
                // res_shift); int64_t quant_res_reg1_64 = rshift_rn_int8((int64_t)res_reg1 *
                // (int64_t)res_mult, res_shift); int64_t quant_res_reg2_64 =
                // rshift_rn_int8((int64_t)res_reg2 * (int64_t)res_mult, res_shift); int64_t
                // quant_res_reg3_64 = rshift_rn_int8((int64_t)res_reg3 * (int64_t)res_mult,
                // res_shift);

                // int64_t quant_res_reg4_64 = rshift_rn_int8((int64_t)res_reg4 * (int64_t)res_mult,
                // res_shift); int64_t quant_res_reg5_64 = rshift_rn_int8((int64_t)res_reg5 *
                // (int64_t)res_mult, res_shift); int64_t quant_res_reg6_64 =
                // rshift_rn_int8((int64_t)res_reg6 * (int64_t)res_mult, res_shift); int64_t
                // quant_res_reg7_64 = rshift_rn_int8((int64_t)res_reg7 * (int64_t)res_mult,
                // res_shift);

                int mval = (1 << (out_bits - 1)) - 1;
                int lower = relu_flag ? 0 : -mval - 1;

                int8_t quant_res_regarr[8];

                quant_res_reg0_alpha = quant_res_reg0_alpha < lower ? lower : quant_res_reg0_alpha;
                quant_res_reg1_alpha = quant_res_reg1_alpha < lower ? lower : quant_res_reg1_alpha;
                quant_res_reg2_alpha = quant_res_reg2_alpha < lower ? lower : quant_res_reg2_alpha;
                quant_res_reg3_alpha = quant_res_reg3_alpha < lower ? lower : quant_res_reg3_alpha;
                quant_res_reg4_alpha = quant_res_reg4_alpha < lower ? lower : quant_res_reg4_alpha;
                quant_res_reg5_alpha = quant_res_reg5_alpha < lower ? lower : quant_res_reg5_alpha;
                quant_res_reg6_alpha = quant_res_reg6_alpha < lower ? lower : quant_res_reg6_alpha;
                quant_res_reg7_alpha = quant_res_reg7_alpha < lower ? lower : quant_res_reg7_alpha;

                quant_res_reg0_alpha = quant_res_reg0_alpha > mval ? mval : quant_res_reg0_alpha;
                quant_res_reg1_alpha = quant_res_reg1_alpha > mval ? mval : quant_res_reg1_alpha;
                quant_res_reg2_alpha = quant_res_reg2_alpha > mval ? mval : quant_res_reg2_alpha;
                quant_res_reg3_alpha = quant_res_reg3_alpha > mval ? mval : quant_res_reg3_alpha;
                quant_res_reg4_alpha = quant_res_reg4_alpha > mval ? mval : quant_res_reg4_alpha;
                quant_res_reg5_alpha = quant_res_reg5_alpha > mval ? mval : quant_res_reg5_alpha;
                quant_res_reg6_alpha = quant_res_reg6_alpha > mval ? mval : quant_res_reg6_alpha;
                quant_res_reg7_alpha = quant_res_reg7_alpha > mval ? mval : quant_res_reg7_alpha;

                quant_res_regarr[0] = quant_res_reg0_alpha;
                quant_res_regarr[1] = quant_res_reg1_alpha;
                quant_res_regarr[2] = quant_res_reg2_alpha;
                quant_res_regarr[3] = quant_res_reg3_alpha;
                quant_res_regarr[4] = quant_res_reg4_alpha;
                quant_res_regarr[5] = quant_res_reg5_alpha;
                quant_res_regarr[6] = quant_res_reg6_alpha;
                quant_res_regarr[7] = quant_res_reg7_alpha;
                // int16_t quant_res_01 = (quant_res_reg1 << 8) | quant_res_reg0;
                // int16_t quant_res_23 = (quant_res_reg3 << 8) | quant_res_reg2;
                // int16_t quant_res_45 = (quant_res_reg5 << 8) | quant_res_reg4;
                // int16_t quant_res_67 = (quant_res_reg7 << 8) | quant_res_reg6;

                if (out_bits == 4) {
                    int8_t quant_res_4bit_0
                        = (quant_res_regarr[0] & 0xF) | (quant_res_regarr[1] << 4);
                    int8_t quant_res_4bit_1
                        = (quant_res_regarr[2] & 0xF) | (quant_res_regarr[3] << 4);
                    int8_t quant_res_4bit_2
                        = (quant_res_regarr[4] & 0xF) | (quant_res_regarr[5] << 4);
                    int8_t quant_res_4bit_3
                        = (quant_res_regarr[6] & 0xF) | (quant_res_regarr[7] << 4);

                    if (actual_row_frag_reg0 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            // int16_t t0 = *((int16_t*)&quant_res_regarr[0]);
                            output_tensor
                                [actual_row_frag_reg0 * out_channel / 2 + actual_col_frag_reg0 / 2]
                                = quant_res_4bit_0; //*((int16_t*)&quant_res_regarr[0]);//quant_res_01;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            // int16_t t1 = *((int16_t*)&quant_res_regarr[2]);

                            output_tensor
                                [actual_row_frag_reg0 * out_channel / 2 + actual_col_frag_reg4 / 2]
                                = quant_res_4bit_1; // *((int16_t*)&quant_res_regarr[2]);//quant_res_45;
                        }
                    }
                    if (actual_row_frag_reg2 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            output_tensor
                                [actual_row_frag_reg2 * out_channel / 2 + actual_col_frag_reg0 / 2]
                                = quant_res_4bit_2; //*((int16_t*)&quant_res_regarr[4]);//quant_res_23;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            output_tensor
                                [actual_row_frag_reg2 * out_channel / 2 + actual_col_frag_reg4 / 2]
                                = quant_res_4bit_3; // *((int16_t*)&quant_res_regarr[6]);//quant_res_67;
                        }
                    }
                } else { // outbits = 8
                    if (actual_row_frag_reg0 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            int16_t t0 = *((int16_t *)&quant_res_regarr[0]);
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg0 * out_channel + actual_col_frag_reg0])
                                = t0; //*((int16_t*)&quant_res_regarr[0]);//quant_res_01;
                                      // output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg0 + 1] = quant_res_regarr1;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            int16_t t1 = *((int16_t *)&quant_res_regarr[2]);

                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg0 * out_channel + actual_col_frag_reg4])
                                = t1; // *((int16_t*)&quant_res_regarr[2]);//quant_res_45;
                        }
                    }
                    if (actual_row_frag_reg2 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg2 * out_channel + actual_col_frag_reg0])
                                = *((int16_t *)&quant_res_regarr[4]); // quant_res_23;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg2 * out_channel + actual_col_frag_reg4])
                                = *((int16_t *)&quant_res_regarr[6]); // quant_res_67;
                        }
                    }
                }
            }
        }
    }
}

template <
    int M_TILED, int N_TILED, int K_TILED, int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K,
    int BLOCK_ROW_WARP_NUM_CHANGE, int BLOCK_COL_WARP_NUM_CHANGE, bool OVERLAP = true>
void lb_turing_indirect_conv_wmma_int8_ldg16_k64_singlebuf_relu(
    char *input_tensor, char *weight_tensor, char *output_tensor, int i_b, int i_c, int i_h,
    int i_w, int k_n, int k_c, int k_h, int k_w, int pad_h, int pad_w, int str_h, int str_w,
    float *bias, bool bias_flag, bool relu_flag, uint8_t bits, float *alpha, void *preComputeA,
    workspace_t *ws)
{

    // TICPU(prepare);
    int out_batch = i_b;
    int out_channel = k_n;

    int out_height = (i_h + pad_h * 2 - k_h) / str_h + 1;
    int out_width = (i_w + pad_w * 2 - k_w) / str_w + 1;

    int GEMM_M = out_batch * out_height * out_width;
    int GEMM_N = out_channel;
    int GEMM_K = i_c * k_h * k_w;

    int GEMM_M_PAD = (GEMM_M + M_TILED - 1) / M_TILED * M_TILED;
    int GEMM_N_PAD = (GEMM_N + N_TILED - 1) / N_TILED * N_TILED;
    int GEMM_K_PAD = (GEMM_K + K_TILED - 1) / K_TILED * K_TILED;

    const int WARP_ROW_WMMA_SZ_CHANGE = M_TILED / (BLOCK_ROW_WARP_NUM_CHANGE * WMMA_16x16_M);
    const int WARP_COL_WMMA_SZ_CHANGE = N_TILED / (BLOCK_COL_WARP_NUM_CHANGE * WMMA_16x16_N);
    const int WMMA_K_STEP_CHANGE = K_TILED / WMMA_16x16_K;

    size_t sharedMM_AB_sz = (M_TILED + N_TILED) * K_TILED;

    // sharedMM_sz = sharedMM_sz < 32768 ? 32768 : sharedMM_sz;
    int matrix_split_M = GEMM_M_PAD / (M_TILED);
    int matrix_split_N = GEMM_N_PAD / (N_TILED);

    preComputeMatixA *preComputeA_ptr = (preComputeMatixA *)preComputeA;

    dim3 gridDim(matrix_split_N, matrix_split_M);

    dim3 blockDim(WARP_SIZE, BLOCK_COL_WARP_NUM_CHANGE, BLOCK_ROW_WARP_NUM_CHANGE);
    cudaFuncSetAttribute(
        lb_turing_indirect_conv_wmma_int8_ldg16_k64_relu<
            WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
            WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE,
            OVERLAP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    // TOCPU(prepare, "preparetime");
    lb_turing_indirect_conv_wmma_int8_ldg16_k64_relu<
        WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
        WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE, OVERLAP>
        <<<gridDim, blockDim, sharedMM_AB_sz, CUDA_WORKSPACE_STREAM(ws)>>>(
            input_tensor, weight_tensor, output_tensor, GEMM_M, GEMM_N, GEMM_K, GEMM_M_PAD,
            GEMM_N_PAD, GEMM_K_PAD, bias, bias_flag, relu_flag, i_b, i_c, i_h, i_w, k_n, k_c, k_h,
            k_w, pad_h, pad_w, str_h, str_w, out_batch, out_channel, out_height, out_width, bits,
            alpha, preComputeA_ptr);
}

template <
    int WARP_ROW_WMMA_SZ_CHANGE, int WARP_COL_WMMA_SZ_CHANGE, int WMMA_K_STEP_CHANGE,
    int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K, int BLOCK_ROW_WARP_NUM_CHANGE,
    int BLOCK_COL_WARP_NUM_CHANGE>
__global__ void lb_turing_indirect_conv_wmma_int8_ldg16_k64_matrixB_relu(
    char *input_tensor, char *weight_tensor, char *output_tensor, int GEMM_M, int GEMM_N,
    int GEMM_K, int GEMM_M_PAD, int GEMM_N_PAD, int GEMM_K_PAD, float *bias, bool bias_flag,
    bool relu_flag, int i_b, int i_c, int i_h, int i_w, int k_n, int k_c, int k_h, int k_w,
    int pad_h, int pad_w, int str_h, int str_w, int out_batch, int out_channel, int out_height,
    int out_width, uint8_t out_bits, float *alphas, preComputeMatixA *preComputeA)
{

    // one
    const int WMMA_ROW_TOTAL_SZ = WARP_ROW_WMMA_SZ_CHANGE * BLOCK_ROW_WARP_NUM_CHANGE;
    const int WMMA_COL_TOTAL_SZ = WARP_COL_WMMA_SZ_CHANGE * BLOCK_COL_WARP_NUM_CHANGE;

    // shared memory pattern:
    // matrix A: [WMMA_ROW_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_M * WMMA_16x16_K]
    // matrix B: [WMMA_COL_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_N * WMMA_16x16_K]

    // there is no need to store MatrixA to the sharemem
    // const int MATRIX_A_SHMEM_STRIDE = 64;//WMMA_16x16_M * WMMA_16x16_K;//WMMA_K_STEP_CHANGE *
    // WMMA_16x16_K + SHMEM_CONFILCT_AVOID;
    const int MATRIX_B_SHMEM_STRIDE = 64; // WMMA_16x16_N * WMMA_16x16_K;

    // Matrix B base address in shared memory
    const int Matrix_B_shmem_offset = 0; // WMMA_ROW_TOTAL_SZ  * MATRIX_A_SHMEM_STRIDE;

    // total warps in a block
    const int WARP_PER_BLOCK_CHANGE = (BLOCK_ROW_WARP_NUM_CHANGE * BLOCK_COL_WARP_NUM_CHANGE);

    // K elements will be processed once at K dimension
    const int K_step = WMMA_K_STEP_CHANGE * WMMA_16x16_K;

    uint32_t warpId = threadIdx.z * blockDim.y + threadIdx.y;
    uint32_t warpIdx = threadIdx.y;
    uint32_t warpIdy = threadIdx.z;

    volatile uint32_t laneId = threadIdx.x;

    uint32_t laneId_0_4 = threadIdx.x & 0x03;
    uint32_t laneId_eighth = threadIdx.x / 4;

    // declare extern shared memory, only will be used if dynamic shared memory used in template
    // function.
    SharedMemory_t<int> shmem;
    const char *shmem_ptr = (char *)shmem.getPointer();

    // Matrix A B block index of one Block
    uint32_t pos_x = blockIdx.x;
    uint32_t pos_y = blockIdx.y;

    {
        int2 my_c_frag[WARP_ROW_WMMA_SZ_CHANGE][WARP_COL_WMMA_SZ_CHANGE][4] = { 0 };

        //   | --- block offset        --------------|  |---   warp offset----------------------- |
        int matrix_A_each_warp_actual_row_base = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ
            + warpId * (WMMA_16x16_M * WARP_ROW_WMMA_SZ_CHANGE);
        int matrix_B_each_warp_actual_col_base = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ
            + warpId * (WMMA_16x16_N * WMMA_COL_TOTAL_SZ / WARP_PER_BLOCK_CHANGE);

        int matrx_B_warp_col = matrix_B_each_warp_actual_col_base
            + laneId_eighth; // laneId_quarter;//laneId_even;//+ laneId_0_15;
        int matrx_A_warp_row = matrix_A_each_warp_actual_row_base
            + laneId_eighth; // laneId_quarter;//laneId_even;//+ laneId_0_15;

        preComputeMatixA MatrixAInfo[WARP_ROW_WMMA_SZ_CHANGE]
                                    [2]; // 2 is for wmma high register and low register
#pragma unroll
        for (int jj = 0; jj < WARP_ROW_WMMA_SZ_CHANGE; jj++) { // 16
            preComputeMatixA *MatrixA_info = preComputeA + matrx_A_warp_row + jj * 16;
            MatrixAInfo[jj][0] = *(preComputeMatixA *)&(__ldg((int4 *)MatrixA_info));
            MatrixAInfo[jj][1] = *(preComputeMatixA *)&(
                __ldg((int4 *)(preComputeA + matrx_A_warp_row + jj * 16 + 8)));
        }

        int i = 0;

        // #pragma unroll
        for (int k_h_index = 0; k_h_index < k_h; k_h_index++) {
            // #pragma unroll
            for (int k_w_index = 0; k_w_index < k_w; k_w_index++) {
                for (int channel_cnt = 0; channel_cnt < i_c; channel_cnt += K_step) {
                    // int4 A_row_frag[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 *
                    // WARP_PER_BLOCK_CHANGE)][K_step / 64];//[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M /
                    // WARP_PER_BLOCK_CHANGE];
                    int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
                    int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];

                    int4 B_row_frag[WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE)]
                                   [K_step / 64]; //[WMMA_COL_TOTAL_SZ * WMMA_16x16_N /
                                                  //WARP_PER_BLOCK_CHANGE];

                    // drag matrix weight, 1 warp 1 line(8x16), each thread drag int
#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
#pragma unroll
                        for (int ii = 0; ii < K_step; ii += 64) {
                            int4 *src = ((
                                int4
                                    *)(weight_tensor + matrx_B_warp_col * GEMM_K_PAD + ii + laneId_0_4 * 16 + jj * 8 * GEMM_K_PAD + (k_h_index * k_w + k_w_index) * i_c + channel_cnt)); // laneId_half * 16
                            B_row_frag[jj][ii / 64] = __ldg(src);
                        }
                    }

#pragma unroll
                    for (int jj = 0; jj < WARP_ROW_WMMA_SZ_CHANGE; jj++) {

                        int32_t image_offset_0 = MatrixAInfo[jj][0].image_offset;
                        int32_t image_offset_1 = MatrixAInfo[jj][1].image_offset;
                        int im_h_0 = MatrixAInfo[jj][0].im_h;
                        int im_w_0 = MatrixAInfo[jj][0].im_w;
                        int im_h_1 = MatrixAInfo[jj][1].im_h;
                        int im_w_1 = MatrixAInfo[jj][1].im_w;
                        bool out_filled_zero_0
                            = MatrixAInfo[jj][0].fill_zero_flag == 1 ? true : false;
                        bool out_filled_zero_1
                            = MatrixAInfo[jj][1].fill_zero_flag == 1 ? true : false;

                        int4 *src_0 = ((
                            int4
                                *)(input_tensor + image_offset_0 + k_h_index * (i_w * i_c) + k_w_index * i_c + channel_cnt + laneId_0_4 * 16)); // laneId_half * 16
                        int4 *src_1 = ((
                            int4
                                *)(input_tensor + image_offset_1 + k_h_index * (i_w * i_c) + k_w_index * i_c + channel_cnt + laneId_0_4 * 16));

                        bool in_window_0 = (im_h_0 + k_h_index >= 0) && (im_h_0 + k_h_index < i_h)
                            && (im_w_0 + k_w_index >= 0) && (im_w_0 + k_w_index < i_w);
                        bool in_window_1 = (im_h_1 + k_h_index >= 0) && (im_h_1 + k_h_index < i_h)
                            && (im_w_1 + k_w_index >= 0) && (im_w_1 + k_w_index < i_w);
                        int4 *src_t_0 = (in_window_0 && (!out_filled_zero_0)) ? src_0 : zero_arr;
                        int4 *src_t_1 = (in_window_1 && (!out_filled_zero_1)) ? src_1 : zero_arr;
                        my_A_frag0[jj] = __ldg(src_t_0);
                        my_A_frag1[jj] = __ldg(src_t_1);
                    }
                    // #pragma unroll
                    //                     for(int jj = 0; jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M /
                    //                     (8 * WARP_PER_BLOCK_CHANGE); jj++){
                    //                         // jj need be filled
                    //                         int32_t image_offset = MatrixAInfo[jj].image_offset;
                    //                         int im_h = MatrixAInfo[jj].im_h;
                    //                         int im_w = MatrixAInfo[jj].im_w;
                    //                         bool out_filled_zero = MatrixAInfo[jj].fill_zero_flag
                    //                         == 1 ? true : false;
                    // #pragma unroll
                    //                         for(int ii = 0; ii < K_step ; ii+=64){

                    //                             int4 * src = ((int4*)(input_tensor + image_offset
                    //                             + k_h_index * (i_w * i_c) + k_w_index * i_c + ii
                    //                             + channel_cnt + laneId_0_4 * 16 )); //laneId_half
                    //                             * 16

                    //                             bool in_window = (im_h + k_h_index >= 0) && (im_h
                    //                             + k_h_index < i_h) && (im_w + k_w_index >= 0) &&
                    //                             (im_w + k_w_index < i_w); int4* src_t =
                    //                             (in_window && (!out_filled_zero)) ? src :
                    //                             zero_arr; A_row_frag[jj][ii/64] =
                    //                             __ldg(src_t);//(in_window && (!out_filled_zero)
                    //                             && (k_h_index < k_h && k_w_index < k_w)) ?
                    //                             __ldg(src) : zero;

                    //                         }
                    //                     }

#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
#pragma unroll
                        for (int ii = 0; ii < WMMA_K_STEP_CHANGE; ii += 4) {
                            int warp_sharedmm_offset = warpId
                                    * (WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_B_SHMEM_STRIDE
                                + ii * MATRIX_B_SHMEM_STRIDE + jj * 8 * MATRIX_B_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_B_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;

                            int4 *dst = ((int4 *)&shmem_ptr
                                             [warp_sharedmm_offset + lane_sharedmm_offset
                                              + Matrix_B_shmem_offset]);

                            *(dst) = B_row_frag[jj][ii / 4];
                        }
                    }

                    __syncthreads();

                    // if (i > 0)
                    {

                        // int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
                        // int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
                        int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
                        int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];

                        for (int k = 0; k < 64; k += 64) {
                            // #pragma unroll
                            //                             for(int row = 0; row <
                            //                             WARP_ROW_WMMA_SZ_CHANGE; row++){
                            //                                 int8_t * a_frag0_ptr =
                            //                                 (int8_t*)&shmem_ptr[(warpIdx *
                            //                                 WARP_ROW_WMMA_SZ_CHANGE + row) * 64 +
                            //                                 k]; int8_t * a_frag1_ptr =
                            //                                 a_frag0_ptr + 8 * 64; my_A_frag0[row]
                            //                                 = ((int4 *)a_frag0_ptr)[threadIdx.x];
                            //                                 my_A_frag1[row] = ((int4
                            //                                 *)a_frag1_ptr)[threadIdx.x];
                            //                             }

#pragma unroll
                            for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {

                                int8_t *b_frag0_ptr = (int8_t *)&shmem_ptr
                                    [col * 16
                                     * MATRIX_B_SHMEM_STRIDE]; // [(warpId * WARP_COL_WMMA_SZ_CHANGE
                                                               // + col) * 64 + k +
                                                               // Matrix_B_shmem_offset];
                                int8_t *b_frag1_ptr = b_frag0_ptr + 8 * MATRIX_B_SHMEM_STRIDE;
                                my_B_frag0[col] = ((int4 *)b_frag0_ptr)[threadIdx.x];
                                my_B_frag1[col] = ((int4 *)b_frag1_ptr)[threadIdx.x];
                            }

#pragma unroll
                            for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
#pragma unroll
                                for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].w,
                                        my_B_frag1[col].w);

                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].w,
                                        my_B_frag1[col].w);

                                    ///*if (my_A_frag0[row].x != 0x01010101|| my_A_frag0[row].y !=
                                    ///0x01010101 || my_A_frag0[row].z != 0x01010101 ||
                                    ///my_A_frag0[row].w != 0x01010101)
                                    //	printf(" %d, %d %d %d \n", my_A_frag0[row].x,
                                    //my_A_frag0[row].y, my_A_frag0[row].z, my_A_frag0[row].w);*/
                                }
                            }
                        }
                        __syncthreads();
                    }

                    // LDG & STS overlap with wmma, now data need to be written to the shared mm, we
                    // assume now the data is ready
                    //  #pragma unroll
                    //              for(int jj = 0; jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (4 *
                    //              WARP_PER_BLOCK_CHANGE); jj++){
                    //                  // jj need be filled
                    //  #pragma unroll
                    //                  for(int ii = 0; ii < WMMA_K_STEP_CHANGE; ii+=4){
                    //                      int warp_sharedmm_offset = warpId * (WMMA_ROW_TOTAL_SZ *
                    //                      WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE)) *
                    //                      MATRIX_A_SHMEM_STRIDE + ii * MATRIX_A_SHMEM_STRIDE + jj
                    //                      * 8 * MATRIX_A_SHMEM_STRIDE; int lane_sharedmm_offset =
                    //                      laneId_0_4 * 16 + laneId_eighth *
                    //                      MATRIX_A_SHMEM_STRIDE;//laneId_even * 16 +
                    //                      laneId_ldg_cacheline * MATRIX_B_SHMEM_STRIDE;//

                    //                     int4 * dst = ((int4*)&shmem_ptr[warp_sharedmm_offset +
                    //                     lane_sharedmm_offset]);

                    //                     *(dst) = A_row_frag[jj][ii/4];
                    //                 }
                    //             }

                    i++;
                }
            }
        }

        //
        //		__syncthreads();
        //
        //
        //		int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
        //		int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
        //		int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
        //		int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];
        //
        //		for (int k = 0; k < 64; k += 64) {
        // #pragma unroll
        //			for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
        //				int8_t * a_frag0_ptr = (int8_t*)&shmem_ptr[(warpIdx *
        //WARP_ROW_WMMA_SZ_CHANGE + row) * 64 + k]; 				int8_t * a_frag1_ptr = a_frag0_ptr + 8 * 64;
        //				my_A_frag0[row] = ((int4 *)a_frag0_ptr)[threadIdx.x];
        //				my_A_frag1[row] = ((int4 *)a_frag1_ptr)[threadIdx.x];
        //			}
        //
        // #pragma unroll
        //			for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
        //				int8_t * b_frag0_ptr = (int8_t*)&shmem_ptr[(warpIdy *
        //WARP_COL_WMMA_SZ_CHANGE + col) * 64 + k + Matrix_B_shmem_offset]; 				int8_t * b_frag1_ptr =
        //b_frag0_ptr + 8 * 64; 				my_B_frag0[col] = ((int4 *)b_frag0_ptr)[threadIdx.x];
        //				my_B_frag1[col] = ((int4 *)b_frag1_ptr)[threadIdx.x];
        //			}
        //
        // #pragma unroll
        //			for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
        // #pragma unroll
        //				for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
        //					my_c_frag[row][col][0] = mma8816(my_c_frag[row][col][0],
        //my_A_frag0[row].x, my_B_frag0[col].x); 					my_c_frag[row][col][0] =
        //mma8816(my_c_frag[row][col][0], my_A_frag0[row].y, my_B_frag0[col].y);
        //					my_c_frag[row][col][0] = mma8816(my_c_frag[row][col][0],
        //my_A_frag0[row].z, my_B_frag0[col].z); 					my_c_frag[row][col][0] =
        //mma8816(my_c_frag[row][col][0], my_A_frag0[row].w, my_B_frag0[col].w);
        //
        //					my_c_frag[row][col][1] = mma8816(my_c_frag[row][col][1],
        //my_A_frag0[row].x, my_B_frag1[col].x); 					my_c_frag[row][col][1] =
        //mma8816(my_c_frag[row][col][1], my_A_frag0[row].y, my_B_frag1[col].y);
        //					my_c_frag[row][col][1] = mma8816(my_c_frag[row][col][1],
        //my_A_frag0[row].z, my_B_frag1[col].z); 					my_c_frag[row][col][1] =
        //mma8816(my_c_frag[row][col][1], my_A_frag0[row].w, my_B_frag1[col].w);
        //
        //					my_c_frag[row][col][2] = mma8816(my_c_frag[row][col][2],
        //my_A_frag1[row].x, my_B_frag0[col].x); 					my_c_frag[row][col][2] =
        //mma8816(my_c_frag[row][col][2], my_A_frag1[row].y, my_B_frag0[col].y);
        //					my_c_frag[row][col][2] = mma8816(my_c_frag[row][col][2],
        //my_A_frag1[row].z, my_B_frag0[col].z); 					my_c_frag[row][col][2] =
        //mma8816(my_c_frag[row][col][2], my_A_frag1[row].w, my_B_frag0[col].w);
        //
        //					my_c_frag[row][col][3] = mma8816(my_c_frag[row][col][3],
        //my_A_frag1[row].x, my_B_frag1[col].x); 					my_c_frag[row][col][3] =
        //mma8816(my_c_frag[row][col][3], my_A_frag1[row].y, my_B_frag1[col].y);
        //					my_c_frag[row][col][3] = mma8816(my_c_frag[row][col][3],
        //my_A_frag1[row].z, my_B_frag1[col].z); 					my_c_frag[row][col][3] =
        //mma8816(my_c_frag[row][col][3], my_A_frag1[row].w, my_B_frag1[col].w);
        //				}
        //			}
        //		}

        float2 alpha0[WARP_COL_WMMA_SZ_CHANGE] = { 0 };
        float2 alpha4[WARP_COL_WMMA_SZ_CHANGE] = { 0 };
#pragma unroll
        for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {
            int channel_id_base = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                + j * WMMA_16x16_N // warp dimension offset
                + laneId % 4 * 2;
            int channel_id_frag_reg0 = (channel_id_base + 0);
            int channel_id_frag_reg4 = (channel_id_base + 8);

            alpha0[j] = __ldg((float2 *)(alphas + channel_id_frag_reg0));
            alpha4[j] = __ldg((float2 *)(alphas + channel_id_frag_reg4));
        }

        float2 bias_channel_reg0[WARP_COL_WMMA_SZ_CHANGE] = { 0 };
        float2 bias_channel_reg4[WARP_COL_WMMA_SZ_CHANGE] = { 0 };

        if (bias_flag) {
#pragma unroll
            for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {

                int channel_id_base
                    = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                    + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                    + j * WMMA_16x16_N // warp dimension offset
                    + laneId % 4 * 2;
                int channel_id_frag_reg0
                    = (channel_id_base + 0); // &~(out_channel - 1);//% out_channel;
                // int channel_id_frag_reg1 = (channel_id_base + 1);// &~(out_channel - 1);// %
                // out_channel;
                //  int channel_id_frag_reg2 = channel_id_base + 0;
                //  int channel_id_frag_reg3 = channel_id_base + 1;

                int channel_id_frag_reg4
                    = (channel_id_base + 8); // &~(out_channel - 1);// % out_channel;
                // int channel_id_frag_reg5 = (channel_id_base + 9);// &~(out_channel - 1);// %
                // out_channel;

                bias_channel_reg0[j] = __ldg((float2 *)(bias + channel_id_frag_reg0));
                bias_channel_reg4[j] = __ldg((float2 *)(bias + channel_id_frag_reg4));
            }
        }

#pragma unroll
        for (int i = 0; i < WARP_ROW_WMMA_SZ_CHANGE; i++) {
#pragma unroll

            for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {

                int channel_id_base
                    = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                    + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                    + j * WMMA_16x16_N // warp dimension offset
                    + laneId % 4 * 2; // thread dimension offset

                int actual_row_base
                    = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ // block dimension offset
                    + warpIdy * WARP_ROW_WMMA_SZ_CHANGE * WMMA_16x16_M // warp dimension offset
                    + i * WMMA_16x16_M // warp dimension offset
                    + laneId / 4; // thread dimension offset

                int actual_col_frag_reg0 = channel_id_base + 0;
                // int actual_col_frag_reg1 = channel_id_base + 1;

                int actual_col_frag_reg4 = channel_id_base + 8;
                // int actual_col_frag_reg5 = channel_id_base + 9;

                int actual_row_frag_reg0 = actual_row_base + 0;
                int actual_row_frag_reg2 = actual_row_base + 8;

                float res_reg0 = my_c_frag[i][j][0].x + bias_channel_reg0[j].x;
                float res_reg1 = my_c_frag[i][j][0].y + bias_channel_reg0[j].y;
                float res_reg2 = my_c_frag[i][j][1].x + bias_channel_reg4[j].x;
                float res_reg3 = my_c_frag[i][j][1].y + bias_channel_reg4[j].y;

                float res_reg4 = my_c_frag[i][j][2].x + bias_channel_reg0[j].x;
                float res_reg5 = my_c_frag[i][j][2].y + bias_channel_reg0[j].y;
                float res_reg6 = my_c_frag[i][j][3].x + bias_channel_reg4[j].x;
                float res_reg7 = my_c_frag[i][j][3].y + bias_channel_reg4[j].y;

                int32_t quant_res_reg0_alpha = (int)rintf((alpha0[j].x * res_reg0));
                int32_t quant_res_reg1_alpha = (int)rintf((alpha0[j].y * res_reg1));
                int32_t quant_res_reg2_alpha = (int)rintf((alpha4[j].x * res_reg2));
                int32_t quant_res_reg3_alpha = (int)rintf((alpha4[j].y * res_reg3));
                int32_t quant_res_reg4_alpha = (int)rintf((alpha0[j].x * res_reg4));
                int32_t quant_res_reg5_alpha = (int)rintf((alpha0[j].y * res_reg5));
                int32_t quant_res_reg6_alpha = (int)rintf((alpha4[j].x * res_reg6));
                int32_t quant_res_reg7_alpha = (int)rintf((alpha4[j].y * res_reg7));

                int mval = (1 << (out_bits - 1)) - 1;
                int lower = relu_flag ? 0 : -mval - 1;

                int8_t quant_res_regarr[8];

                quant_res_reg0_alpha = quant_res_reg0_alpha < lower ? lower : quant_res_reg0_alpha;
                quant_res_reg1_alpha = quant_res_reg1_alpha < lower ? lower : quant_res_reg1_alpha;
                quant_res_reg2_alpha = quant_res_reg2_alpha < lower ? lower : quant_res_reg2_alpha;
                quant_res_reg3_alpha = quant_res_reg3_alpha < lower ? lower : quant_res_reg3_alpha;
                quant_res_reg4_alpha = quant_res_reg4_alpha < lower ? lower : quant_res_reg4_alpha;
                quant_res_reg5_alpha = quant_res_reg5_alpha < lower ? lower : quant_res_reg5_alpha;
                quant_res_reg6_alpha = quant_res_reg6_alpha < lower ? lower : quant_res_reg6_alpha;
                quant_res_reg7_alpha = quant_res_reg7_alpha < lower ? lower : quant_res_reg7_alpha;

                quant_res_reg0_alpha = quant_res_reg0_alpha > mval ? mval : quant_res_reg0_alpha;
                quant_res_reg1_alpha = quant_res_reg1_alpha > mval ? mval : quant_res_reg1_alpha;
                quant_res_reg2_alpha = quant_res_reg2_alpha > mval ? mval : quant_res_reg2_alpha;
                quant_res_reg3_alpha = quant_res_reg3_alpha > mval ? mval : quant_res_reg3_alpha;
                quant_res_reg4_alpha = quant_res_reg4_alpha > mval ? mval : quant_res_reg4_alpha;
                quant_res_reg5_alpha = quant_res_reg5_alpha > mval ? mval : quant_res_reg5_alpha;
                quant_res_reg6_alpha = quant_res_reg6_alpha > mval ? mval : quant_res_reg6_alpha;
                quant_res_reg7_alpha = quant_res_reg7_alpha > mval ? mval : quant_res_reg7_alpha;

                quant_res_regarr[0] = quant_res_reg0_alpha;
                quant_res_regarr[1] = quant_res_reg1_alpha;
                quant_res_regarr[2] = quant_res_reg2_alpha;
                quant_res_regarr[3] = quant_res_reg3_alpha;
                quant_res_regarr[4] = quant_res_reg4_alpha;
                quant_res_regarr[5] = quant_res_reg5_alpha;
                quant_res_regarr[6] = quant_res_reg6_alpha;
                quant_res_regarr[7] = quant_res_reg7_alpha;

                if (out_bits == 4) {
                    int8_t quant_res_4bit_0
                        = (quant_res_regarr[0] & 0xF) | (quant_res_regarr[1] << 4);
                    int8_t quant_res_4bit_1
                        = (quant_res_regarr[2] & 0xF) | (quant_res_regarr[3] << 4);
                    int8_t quant_res_4bit_2
                        = (quant_res_regarr[4] & 0xF) | (quant_res_regarr[5] << 4);
                    int8_t quant_res_4bit_3
                        = (quant_res_regarr[6] & 0xF) | (quant_res_regarr[7] << 4);

                    if (actual_row_frag_reg0 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            // int16_t t0 = *((int16_t*)&quant_res_regarr[0]);
                            output_tensor
                                [actual_row_frag_reg0 * out_channel / 2 + actual_col_frag_reg0 / 2]
                                = quant_res_4bit_0; //*((int16_t*)&quant_res_regarr[0]);//quant_res_01;
                                                    // output_tensor[actual_row_frag_reg0 *
                                                    // out_channel + actual_col_frag_reg0] =
                                                    // quant_res_regarr0;
                                                    // output_tensor[actual_row_frag_reg0 *
                                                    // out_channel + actual_col_frag_reg0 + 1] =
                                                    // quant_res_regarr1;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            // int16_t t1 = *((int16_t*)&quant_res_regarr[2]);

                            output_tensor
                                [actual_row_frag_reg0 * out_channel / 2 + actual_col_frag_reg4 / 2]
                                = quant_res_4bit_1; // *((int16_t*)&quant_res_regarr[2]);//quant_res_45;
                                                    // output_tensor[actual_row_frag_reg0 *
                                                    // out_channel + actual_col_frag_reg4] =
                                                    // quant_res_regarr2;
                                                    // output_tensor[actual_row_frag_reg0 *
                                                    // out_channel + actual_col_frag_reg4 + 1] =
                                                    // quant_res_regarr3;
                                                    // //output_tensor[actual_row_frag_reg0 *
                                                    // out_channel + actual_col_frag_reg4 + 1] =
                                                    // quant_res_reg5;
                        }
                    }
                    if (actual_row_frag_reg2 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            output_tensor
                                [actual_row_frag_reg2 * out_channel / 2 + actual_col_frag_reg0 / 2]
                                = quant_res_4bit_2; //*((int16_t*)&quant_res_regarr[4]);//quant_res_23;
                                                    // output_tensor[actual_row_frag_reg2 *
                                                    // out_channel + actual_col_frag_reg0] =
                                                    // quant_res_regarr4;
                                                    // output_tensor[actual_row_frag_reg2 *
                                                    // out_channel + actual_col_frag_reg0 + 1] =
                                                    // quant_res_regarr5;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            output_tensor
                                [actual_row_frag_reg2 * out_channel / 2 + actual_col_frag_reg4 / 2]
                                = quant_res_4bit_3; // *((int16_t*)&quant_res_regarr[6]);//quant_res_67;
                                                    // output_tensor[actual_row_frag_reg2 *
                                                    // out_channel + actual_col_frag_reg4] =
                                                    // quant_res_regarr6;
                                                    // output_tensor[actual_row_frag_reg2 *
                                                    // out_channel + actual_col_frag_reg4 + 1] =
                                                    // quant_res_regarr7;
                        }
                    }
                } else { // outbits = 8
                    if (actual_row_frag_reg0 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            int16_t t0 = *((int16_t *)&quant_res_regarr[0]);
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg0 * out_channel + actual_col_frag_reg0])
                                = t0; //*((int16_t*)&quant_res_regarr[0]);//quant_res_01;
                                      // output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg0] = quant_res_regarr0;
                                      // output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg0 + 1] = quant_res_regarr1;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            int16_t t1 = *((int16_t *)&quant_res_regarr[2]);

                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg0 * out_channel + actual_col_frag_reg4])
                                = t1; // *((int16_t*)&quant_res_regarr[2]);//quant_res_45;
                                      // output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg4] = quant_res_regarr2;
                                      // output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg4 + 1] = quant_res_regarr3;
                                      // //output_tensor[actual_row_frag_reg0 * out_channel +
                                      // actual_col_frag_reg4 + 1] = quant_res_reg5;
                        }
                    }
                    if (actual_row_frag_reg2 < out_batch * out_height * out_width) {
                        if (actual_col_frag_reg0 < out_channel) {
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg2 * out_channel + actual_col_frag_reg0])
                                = *((int16_t *)&quant_res_regarr
                                        [4]); // quant_res_23;
                                              // output_tensor[actual_row_frag_reg2 * out_channel +
                                              // actual_col_frag_reg0] = quant_res_regarr4;
                                              // output_tensor[actual_row_frag_reg2 * out_channel +
                                              // actual_col_frag_reg0 + 1] = quant_res_regarr5;
                        }
                        if (actual_col_frag_reg4 < out_channel) {
                            *((int16_t *)&output_tensor
                                  [actual_row_frag_reg2 * out_channel + actual_col_frag_reg4])
                                = *((int16_t *)&quant_res_regarr
                                        [6]); // quant_res_67;
                                              // output_tensor[actual_row_frag_reg2 * out_channel +
                                              // actual_col_frag_reg4] = quant_res_regarr6;
                                              // output_tensor[actual_row_frag_reg2 * out_channel +
                                              // actual_col_frag_reg4 + 1] = quant_res_regarr7;
                        }
                    }
                }
            }
        }
    }
}

template <
    int M_TILED, int N_TILED, int K_TILED, int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K,
    int BLOCK_ROW_WARP_NUM_CHANGE, int BLOCK_COL_WARP_NUM_CHANGE>
void lb_turing_indirect_conv_wmma_int8_ldg16_k64_matrixB_singlebuf_relu(
    char *input_tensor, char *weight_tensor, char *output_tensor, int i_b, int i_c, int i_h,
    int i_w, int k_n, int k_c, int k_h, int k_w, int pad_h, int pad_w, int str_h, int str_w,
    float *bias, bool bias_flag, bool relu_flag, uint8_t bits, float *alpha, void *preComputeA,
    workspace_t *ws)
{

    int out_batch = i_b;
    int out_channel = k_n;

    int out_height = (i_h + pad_h * 2 - k_h) / str_h + 1;
    int out_width = (i_w + pad_w * 2 - k_w) / str_w + 1;

    int GEMM_M = out_batch * out_height * out_width;
    int GEMM_N = out_channel;
    int GEMM_K = i_c * k_h * k_w;

    int GEMM_M_PAD = (GEMM_M + M_TILED - 1) / M_TILED * M_TILED;
    int GEMM_N_PAD = (GEMM_N + N_TILED - 1) / N_TILED * N_TILED;
    int GEMM_K_PAD = (GEMM_K + K_TILED - 1) / K_TILED * K_TILED;

    const int WARP_ROW_WMMA_SZ_CHANGE = M_TILED / (BLOCK_ROW_WARP_NUM_CHANGE * WMMA_16x16_M);
    const int WARP_COL_WMMA_SZ_CHANGE = N_TILED / (BLOCK_COL_WARP_NUM_CHANGE * WMMA_16x16_N);
    const int WMMA_K_STEP_CHANGE = K_TILED / WMMA_16x16_K;

    size_t sharedMM_AB_sz = N_TILED * K_TILED; //(M_TILED + N_TILED) * K_TILED;

    // sharedMM_sz = sharedMM_sz < 32768 ? 32768 : sharedMM_sz;
    int matrix_split_M = GEMM_M_PAD / (M_TILED);
    int matrix_split_N = GEMM_N_PAD / (N_TILED);

    preComputeMatixA *preComputeA_ptr = (preComputeMatixA *)preComputeA;

    dim3 gridDim(matrix_split_N, matrix_split_M);
    dim3 blockDim(WARP_SIZE, BLOCK_COL_WARP_NUM_CHANGE, BLOCK_ROW_WARP_NUM_CHANGE);
    cudaFuncSetAttribute(
        lb_turing_indirect_conv_wmma_int8_ldg16_k64_matrixB_relu<
            WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
            WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    lb_turing_indirect_conv_wmma_int8_ldg16_k64_matrixB_relu<
        WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
        WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE>
        <<<gridDim, blockDim, sharedMM_AB_sz, CUDA_WORKSPACE_STREAM(ws)>>>(
            input_tensor, weight_tensor, output_tensor, GEMM_M, GEMM_N, GEMM_K, GEMM_M_PAD,
            GEMM_N_PAD, GEMM_K_PAD, bias, bias_flag, relu_flag, i_b, i_c, i_h, i_w, k_n, k_c, k_h,
            k_w, pad_h, pad_w, str_h, str_w, out_batch, out_channel, out_height, out_width, bits,
            alpha, preComputeA_ptr);
}

template <
    int WARP_ROW_WMMA_SZ_CHANGE, int WARP_COL_WMMA_SZ_CHANGE, int WMMA_K_STEP_CHANGE,
    int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K, int BLOCK_ROW_WARP_NUM_CHANGE,
    int BLOCK_COL_WARP_NUM_CHANGE, bool overlap>
__global__ void lb_turing_indirect_conv_wmma_int8_ldg16_k64_spilt3x3_relu(
    char *input_tensor, char *weight_tensor, int *output_tensor_tmp, int GEMM_M, int GEMM_N,
    int GEMM_K, int GEMM_M_PAD, int GEMM_N_PAD, int GEMM_K_PAD, float *bias, bool bias_flag,
    bool relu_flag, int i_b, int i_c, int i_h, int i_w, int k_n, int k_c, int k_h, int k_w,
    int pad_h, int pad_w, int str_h, int str_w, int out_batch, int out_channel, int out_height,
    int out_width, uint8_t out_bits, float *alphas, preComputeMatixA *preComputeA)

{
    uint32_t split_k_h_index = blockIdx.z / 3;
    uint32_t split_k_w_index = blockIdx.z - split_k_h_index * 3;
    // one
    const int WMMA_ROW_TOTAL_SZ = WARP_ROW_WMMA_SZ_CHANGE * BLOCK_ROW_WARP_NUM_CHANGE;
    const int WMMA_COL_TOTAL_SZ = WARP_COL_WMMA_SZ_CHANGE * BLOCK_COL_WARP_NUM_CHANGE;

    // shared memory pattern:
    // matrix A: [WMMA_ROW_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_M * WMMA_16x16_K]
    // matrix B: [WMMA_COL_TOTAL_SZ][WMMA_K_STEP_CHANGE][WMMA_16x16_N * WMMA_16x16_K]
    const int MATRIX_A_SHMEM_STRIDE = 64; // WMMA_16x16_M * WMMA_16x16_K;//WMMA_K_STEP_CHANGE *
                                          // WMMA_16x16_K + SHMEM_CONFILCT_AVOID;
    const int MATRIX_B_SHMEM_STRIDE = 64; // WMMA_16x16_N * WMMA_16x16_K;

    // Matrix B base address in shared memory
    const int Matrix_B_shmem_offset = WMMA_ROW_TOTAL_SZ * WMMA_16x16_M * MATRIX_A_SHMEM_STRIDE;

    // total warps in a block
    const int WARP_PER_BLOCK_CHANGE = (BLOCK_ROW_WARP_NUM_CHANGE * BLOCK_COL_WARP_NUM_CHANGE);

    // K elements will be processed once at K dimension
    const int K_step = WMMA_K_STEP_CHANGE * WMMA_16x16_K;

    uint32_t warpId = threadIdx.z * blockDim.y + threadIdx.y;
    uint32_t warpIdx = threadIdx.y;
    uint32_t warpIdy = threadIdx.z;

    volatile uint32_t laneId = threadIdx.x;

    uint32_t laneId_even = threadIdx.x / 2;
    uint32_t laneId_ldg_cacheline = threadIdx.x & 1;

    uint32_t laneId_0_8 = threadIdx.x & 0x07;
    // // if a thread is in 0-15 in a warp, will be 0; else will be 1
    uint32_t laneId_quarter = threadIdx.x / 8;

    uint32_t laneId_0_4 = threadIdx.x & 0x03;
    uint32_t laneId_eighth = threadIdx.x / 4;

    //  uint32_t laneId_0_15 = threadIdx.x & 0x0F;
    // // // if a thread is in 0-15 in a warp, will be 0; else will be 1
    //  uint32_t laneId_half = threadIdx.x > 0x0F ? 1 : 0;

    // declare extern shared memory, only will be used if dynamic shared memory used in template
    // function.
    SharedMemory_t<int> shmem;
    const char *shmem_ptr = (char *)shmem.getPointer();
    // extern __shared__ char shmem_ptr[];

    // Matrix A B block index of one Block
    uint32_t pos_x = blockIdx.x;
    uint32_t pos_y = blockIdx.y;

    {
        int2 my_c_frag[WARP_ROW_WMMA_SZ_CHANGE][WARP_COL_WMMA_SZ_CHANGE][4] = { 0 };

        //   | --- block offset        --------------|  |---   warp offset----------------------- |
        int matrix_A_each_warp_actual_row_base = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ
            + warpId * (WMMA_16x16_M * WMMA_ROW_TOTAL_SZ / WARP_PER_BLOCK_CHANGE);
        int matrix_B_each_warp_actual_col_base = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ
            + warpId * (WMMA_16x16_N * WMMA_COL_TOTAL_SZ / WARP_PER_BLOCK_CHANGE);

        int matrx_B_warp_col
            = matrix_B_each_warp_actual_col_base + laneId_eighth; // laneId_even;//+ laneId_0_15;
        int matrx_A_warp_row
            = matrix_A_each_warp_actual_row_base + laneId_eighth; // laneId_even;//+ laneId_0_15;

        preComputeMatixA
            MatrixAInfo[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE * 4)]; // 16
#pragma unroll
        for (int jj = 0; jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
             jj++) { // 16
            preComputeMatixA *MatrixA_info = preComputeA + matrx_A_warp_row + jj * 8;
            MatrixAInfo[jj] = *(preComputeMatixA *)&(__ldg((int4 *)MatrixA_info));
        }

        int i = 0;

        // #pragma unroll
        // for (int k_h_index = 0; k_h_index < k_h; k_h_index++)
        int k_h_index = split_k_h_index;
        {
            // #pragma unroll
            // for (int k_w_index = 0; k_w_index < k_w; k_w_index++)
            int k_w_index = split_k_w_index;
            {
                for (int channel_cnt = 0; channel_cnt < i_c; channel_cnt += K_step) {
                    int4 A_row_frag
                        [WMMA_ROW_TOTAL_SZ * WMMA_16x16_M
                         / (8 * WARP_PER_BLOCK_CHANGE)]; //[WMMA_ROW_TOTAL_SZ * WMMA_16x16_M /
                                                         //WARP_PER_BLOCK_CHANGE];
                    int4 B_row_frag
                        [WMMA_COL_TOTAL_SZ * WMMA_16x16_N
                         / (8 * WARP_PER_BLOCK_CHANGE)]; //[WMMA_COL_TOTAL_SZ * WMMA_16x16_N /
                                                         //WARP_PER_BLOCK_CHANGE];

                    // drag matrix weight, 1 warp 1 line(8x16), each thread drag int
#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
                        int4 *src = ((
                            int4
                                *)(weight_tensor + matrx_B_warp_col * GEMM_K_PAD + laneId_0_4 * 16 + jj * 8 * GEMM_K_PAD + (k_h_index * k_w + k_w_index) * i_c + channel_cnt)); // laneId_half * 16
                        B_row_frag[jj] = __ldg(src);
                    }

#pragma unroll
                    for (int jj = 0;
                         jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                         jj++) {
                        // jj need be filled
                        int32_t image_offset = MatrixAInfo[jj].image_offset;
                        int im_h = MatrixAInfo[jj].im_h;
                        int im_w = MatrixAInfo[jj].im_w;
                        bool out_filled_zero = MatrixAInfo[jj].fill_zero_flag == 1 ? true : false;

                        int4 *src = ((
                            int4
                                *)(input_tensor + image_offset + k_h_index * (i_w * i_c) + k_w_index * i_c + channel_cnt + laneId_0_4 * 16)); // laneId_half * 16

                        bool in_window = (im_h + k_h_index >= 0) && (im_h + k_h_index < i_h)
                            && (im_w + k_w_index >= 0) && (im_w + k_w_index < i_w);
                        int4 *src_t = (in_window && (!out_filled_zero)) ? src : zero_arr;
                        A_row_frag[jj]
                            = __ldg(src_t); //(in_window && (!out_filled_zero) && (k_h_index < k_h
                                            //&& k_w_index < k_w)) ? __ldg(src) : zero;
                    }
                    if (!overlap) {
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled
                            int warp_sharedmm_offset = warpId
                                    * (WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_B_SHMEM_STRIDE
                                + jj * 8 * MATRIX_B_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_B_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;

                            int4 *dst = ((int4 *)(&shmem_ptr
                                                      [warp_sharedmm_offset + lane_sharedmm_offset
                                                       + Matrix_B_shmem_offset]));

                            *(dst) = B_row_frag[jj];
                        }

                        // LDG & STS overlap with wmma, now data need to be written to the shared
                        // mm, we assume now the data is ready
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled

                            int warp_sharedmm_offset = warpId
                                    * (WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_A_SHMEM_STRIDE
                                + jj * 8 * MATRIX_A_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_A_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;//

                            int4 *dst = ((
                                int4 *)(&shmem_ptr[warp_sharedmm_offset + lane_sharedmm_offset]));

                            *(dst) = A_row_frag[jj];
                        }
                    }

                    __syncthreads();

                    int cnt = overlap ? 0 : -1;
                    if (i > cnt) {

                        int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
                        int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
                        int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
                        int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];
#pragma unroll
                        for (int k = 0; k < 64; k += 64) {

#pragma unroll
                            for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
                                int warp_offset = (warpIdy * WARP_ROW_WMMA_SZ_CHANGE + row)
                                        * WMMA_16x16_M * MATRIX_A_SHMEM_STRIDE
                                    + k;
                                int lane_offset
                                    = laneId_0_4 * 16 + laneId_eighth * MATRIX_A_SHMEM_STRIDE;
                                int8_t *a_frag0_ptr = (int8_t *)shmem_ptr;
                                int8_t *a_frag1_ptr = a_frag0_ptr + 8 * MATRIX_A_SHMEM_STRIDE;
                                my_A_frag0[row]
                                    = *((int4 *)(&a_frag0_ptr[warp_offset + lane_offset]));
                                my_A_frag1[row]
                                    = *((int4 *)(&a_frag1_ptr[warp_offset + lane_offset]));
                            }

#pragma unroll
                            for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                                int warp_offset = (warpIdx * WARP_COL_WMMA_SZ_CHANGE + col)
                                        * WMMA_16x16_N * MATRIX_B_SHMEM_STRIDE
                                    + k;
                                int lane_offset
                                    = laneId_0_4 * 16 + laneId_eighth * MATRIX_B_SHMEM_STRIDE;
                                int8_t *b_frag0_ptr = (int8_t *)shmem_ptr + Matrix_B_shmem_offset;
                                int8_t *b_frag1_ptr = b_frag0_ptr + 8 * MATRIX_B_SHMEM_STRIDE;
                                my_B_frag0[col]
                                    = *((int4 *)(&b_frag0_ptr[warp_offset + lane_offset]));
                                my_B_frag1[col]
                                    = *((int4 *)(&b_frag1_ptr[warp_offset + lane_offset]));
                            }

#pragma unroll
                            for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
#pragma unroll
                                for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][0] = mma8816(
                                        my_c_frag[row][col][0], my_A_frag0[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][1] = mma8816(
                                        my_c_frag[row][col][1], my_A_frag0[row].w,
                                        my_B_frag1[col].w);

                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].x,
                                        my_B_frag0[col].x);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].y,
                                        my_B_frag0[col].y);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].z,
                                        my_B_frag0[col].z);
                                    my_c_frag[row][col][2] = mma8816(
                                        my_c_frag[row][col][2], my_A_frag1[row].w,
                                        my_B_frag0[col].w);

                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].x,
                                        my_B_frag1[col].x);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].y,
                                        my_B_frag1[col].y);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].z,
                                        my_B_frag1[col].z);
                                    my_c_frag[row][col][3] = mma8816(
                                        my_c_frag[row][col][3], my_A_frag1[row].w,
                                        my_B_frag1[col].w);
                                }
                            }
                        }
                        __syncthreads();
                    }

                    if (overlap) {
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled
                            int warp_sharedmm_offset = warpId
                                    * (WMMA_COL_TOTAL_SZ * WMMA_16x16_N / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_B_SHMEM_STRIDE
                                + jj * 8 * MATRIX_B_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_B_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;

                            int4 *dst = ((int4 *)(&shmem_ptr
                                                      [warp_sharedmm_offset + lane_sharedmm_offset
                                                       + Matrix_B_shmem_offset]));

                            *(dst) = B_row_frag[jj];
                        }

                        // LDG & STS overlap with wmma, now data need to be written to the shared
                        // mm, we assume now the data is ready
#pragma unroll
                        for (int jj = 0;
                             jj < WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (8 * WARP_PER_BLOCK_CHANGE);
                             jj++) {
                            // jj need be filled

                            int warp_sharedmm_offset = warpId
                                    * (WMMA_ROW_TOTAL_SZ * WMMA_16x16_M / (WARP_PER_BLOCK_CHANGE))
                                    * MATRIX_A_SHMEM_STRIDE
                                + jj * 8 * MATRIX_A_SHMEM_STRIDE;
                            int lane_sharedmm_offset = laneId_0_4 * 16
                                + laneId_eighth * MATRIX_A_SHMEM_STRIDE; // laneId_even * 16 +
                                                                         // laneId_ldg_cacheline *
                                                                         // MATRIX_B_SHMEM_STRIDE;//

                            int4 *dst = ((
                                int4 *)(&shmem_ptr[warp_sharedmm_offset + lane_sharedmm_offset]));

                            *(dst) = A_row_frag[jj];
                        }
                    }

                    i++;
                }
            }
        }

        if (overlap) {
            int4 my_A_frag0[WARP_ROW_WMMA_SZ_CHANGE];
            int4 my_A_frag1[WARP_ROW_WMMA_SZ_CHANGE];
            int4 my_B_frag0[WARP_COL_WMMA_SZ_CHANGE];
            int4 my_B_frag1[WARP_COL_WMMA_SZ_CHANGE];

            __syncthreads();
#pragma unroll
            for (int k = 0; k < 64; k += 64) {

#pragma unroll
                for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
                    int warp_offset = (warpIdy * WARP_ROW_WMMA_SZ_CHANGE + row) * WMMA_16x16_M
                            * MATRIX_A_SHMEM_STRIDE
                        + k;
                    int lane_offset = laneId_0_4 * 16 + laneId_eighth * MATRIX_A_SHMEM_STRIDE;
                    int8_t *a_frag0_ptr = (int8_t *)shmem_ptr;
                    int8_t *a_frag1_ptr = a_frag0_ptr + 8 * MATRIX_A_SHMEM_STRIDE;
                    my_A_frag0[row] = *((int4 *)(&a_frag0_ptr[warp_offset + lane_offset]));
                    my_A_frag1[row] = *((int4 *)(&a_frag1_ptr[warp_offset + lane_offset]));
                }

#pragma unroll
                for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                    int warp_offset = (warpIdx * WARP_COL_WMMA_SZ_CHANGE + col) * WMMA_16x16_N
                            * MATRIX_B_SHMEM_STRIDE
                        + k;
                    int lane_offset = laneId_0_4 * 16 + laneId_eighth * MATRIX_B_SHMEM_STRIDE;
                    int8_t *b_frag0_ptr = (int8_t *)shmem_ptr + Matrix_B_shmem_offset;
                    int8_t *b_frag1_ptr = b_frag0_ptr + 8 * MATRIX_B_SHMEM_STRIDE;
                    my_B_frag0[col] = *((int4 *)(&b_frag0_ptr[warp_offset + lane_offset]));
                    my_B_frag1[col] = *((int4 *)(&b_frag1_ptr[warp_offset + lane_offset]));
                }

#pragma unroll
                for (int row = 0; row < WARP_ROW_WMMA_SZ_CHANGE; row++) {
#pragma unroll
                    for (int col = 0; col < WARP_COL_WMMA_SZ_CHANGE; col++) {
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].x, my_B_frag0[col].x);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].y, my_B_frag0[col].y);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].z, my_B_frag0[col].z);
                        my_c_frag[row][col][0]
                            = mma8816(my_c_frag[row][col][0], my_A_frag0[row].w, my_B_frag0[col].w);

                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].x, my_B_frag1[col].x);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].y, my_B_frag1[col].y);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].z, my_B_frag1[col].z);
                        my_c_frag[row][col][1]
                            = mma8816(my_c_frag[row][col][1], my_A_frag0[row].w, my_B_frag1[col].w);

                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].x, my_B_frag0[col].x);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].y, my_B_frag0[col].y);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].z, my_B_frag0[col].z);
                        my_c_frag[row][col][2]
                            = mma8816(my_c_frag[row][col][2], my_A_frag1[row].w, my_B_frag0[col].w);

                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].x, my_B_frag1[col].x);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].y, my_B_frag1[col].y);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].z, my_B_frag1[col].z);
                        my_c_frag[row][col][3]
                            = mma8816(my_c_frag[row][col][3], my_A_frag1[row].w, my_B_frag1[col].w);
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < WARP_ROW_WMMA_SZ_CHANGE; i++) {
#pragma unroll

            for (int j = 0; j < WARP_COL_WMMA_SZ_CHANGE; j++) {

                int channel_id_base
                    = pos_x * WMMA_16x16_N * WMMA_COL_TOTAL_SZ // block dimension offset
                    + warpIdx * WARP_COL_WMMA_SZ_CHANGE * WMMA_16x16_N // warp dimension offset
                    + j * WMMA_16x16_N // warp dimension offset
                    + laneId % 4 * 2; // thread dimension offset

                int actual_row_base
                    = pos_y * WMMA_16x16_M * WMMA_ROW_TOTAL_SZ // block dimension offset
                    + warpIdy * WARP_ROW_WMMA_SZ_CHANGE * WMMA_16x16_M // warp dimension offset
                    + i * WMMA_16x16_M // warp dimension offset
                    + laneId / 4; // thread dimension offset

                int actual_col_frag_reg0 = channel_id_base + 0;
                // int actual_col_frag_reg1 = channel_id_base + 1;

                int actual_col_frag_reg4 = channel_id_base + 8;
                // int actual_col_frag_reg5 = channel_id_base + 9;

                int actual_row_frag_reg0 = actual_row_base + 0;
                int actual_row_frag_reg2 = actual_row_base + 8;
                int32_t res_reg0 = my_c_frag[i][j][0].x; // +bias_channel_reg0[j].x;
                int32_t res_reg1 = my_c_frag[i][j][0].y; // +bias_channel_reg0[j].y;
                int32_t res_reg2 = my_c_frag[i][j][1].x; // +bias_channel_reg4[j].x;
                int32_t res_reg3 = my_c_frag[i][j][1].y; // +bias_channel_reg4[j].y;

                int32_t res_reg4 = my_c_frag[i][j][2].x; // +bias_channel_reg0[j].x;
                int32_t res_reg5 = my_c_frag[i][j][2].y; // +bias_channel_reg0[j].y;
                int32_t res_reg6 = my_c_frag[i][j][3].x; // +bias_channel_reg4[j].x;
                int32_t res_reg7 = my_c_frag[i][j][3].y; // +bias_channel_reg4[j].y;

                /*float2 quant_res_reg[4];

                quant_res_reg[0].x = ((alpha0[j].x * (float)res_reg0));
                quant_res_reg[0].y = ((alpha0[j].y * (float)res_reg1));
                quant_res_reg[1].x = ((alpha4[j].x * (float)res_reg2));
                quant_res_reg[1].y = ((alpha4[j].y * (float)res_reg3));
                quant_res_reg[2].x = ((alpha0[j].x * (float)res_reg4));
                quant_res_reg[2].y = ((alpha0[j].y * (float)res_reg5));
                quant_res_reg[3].x = ((alpha4[j].x * (float)res_reg6));
                quant_res_reg[3].y = ((alpha4[j].y * (float)res_reg7));*/

                if (actual_row_frag_reg0 < out_batch * out_height * out_width) {
                    if (actual_col_frag_reg0 < out_channel) {
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg0 * out_channel
                                + actual_col_frag_reg0,
                            res_reg0);
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg0 * out_channel
                                + actual_col_frag_reg0 + 1,
                            res_reg1);
                    }
                    if (actual_col_frag_reg4 < out_channel) {
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg0 * out_channel
                                + actual_col_frag_reg4,
                            res_reg2);
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg0 * out_channel
                                + actual_col_frag_reg4 + 1,
                            res_reg3);
                    }
                }
                if (actual_row_frag_reg2 < out_batch * out_height * out_width) {
                    if (actual_col_frag_reg0 < out_channel) {
                        //*(float2*)(&tmp[actual_row_frag_reg2 * out_channel +
                        //actual_col_frag_reg0]) = quant_res_reg[2];
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg2 * out_channel
                                + actual_col_frag_reg0,
                            res_reg4);
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg2 * out_channel
                                + actual_col_frag_reg0 + 1,
                            res_reg5);
                    }
                    if (actual_col_frag_reg4 < out_channel) {
                        //*(float2*)(&tmp[actual_row_frag_reg2 * out_channel +
                        //actual_col_frag_reg4]) = quant_res_reg[3];
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg2 * out_channel
                                + actual_col_frag_reg4,
                            res_reg6);
                        atomicAdd(
                            output_tensor_tmp + actual_row_frag_reg2 * out_channel
                                + actual_col_frag_reg4 + 1,
                            res_reg7);
                    }
                }
            }
        }
    }
}

template <
    int M_TILED, int N_TILED, int K_TILED, int WMMA_16x16_M, int WMMA_16x16_N, int WMMA_16x16_K,
    int BLOCK_ROW_WARP_NUM_CHANGE, int BLOCK_COL_WARP_NUM_CHANGE, bool OVERLAP>
void lb_turing_indirect_conv_wmma_int8_ldg16_k64_split3x3_singlebuf_relu(
    char *input_tensor, char *weight_tensor, char *output_tensor, int *tmp, int i_b, int i_c,
    int i_h, int i_w, int k_n, int k_c, int k_h, int k_w, int pad_h, int pad_w, int str_h,
    int str_w, float *bias, bool bias_flag, bool relu_flag, uint8_t bits, float *alpha,
    void *preComputeA, workspace_t *ws)
{

    int out_batch = i_b;
    int out_channel = k_n;

    int out_height = (i_h + pad_h * 2 - k_h) / str_h + 1;
    int out_width = (i_w + pad_w * 2 - k_w) / str_w + 1;

    int GEMM_M = out_batch * out_height * out_width;
    int GEMM_N = out_channel;
    int GEMM_K = i_c * k_h * k_w;

    int GEMM_M_PAD = (GEMM_M + M_TILED - 1) / M_TILED * M_TILED;
    int GEMM_N_PAD = (GEMM_N + N_TILED - 1) / N_TILED * N_TILED;
    int GEMM_K_PAD = (GEMM_K + K_TILED - 1) / K_TILED * K_TILED;

    const int WARP_ROW_WMMA_SZ_CHANGE = M_TILED / (BLOCK_ROW_WARP_NUM_CHANGE * WMMA_16x16_M);
    const int WARP_COL_WMMA_SZ_CHANGE = N_TILED / (BLOCK_COL_WARP_NUM_CHANGE * WMMA_16x16_N);
    const int WMMA_K_STEP_CHANGE = K_TILED / WMMA_16x16_K;

    size_t sharedMM_AB_sz = (M_TILED + N_TILED) * K_TILED;

    // sharedMM_sz = sharedMM_sz < 32768 ? 32768 : sharedMM_sz;
    int matrix_split_M = GEMM_M_PAD / (M_TILED);
    int matrix_split_N = GEMM_N_PAD / (N_TILED);

    preComputeMatixA *preComputeA_ptr = (preComputeMatixA *)preComputeA;

    dim3 gridDim(matrix_split_N, matrix_split_M, k_h * k_w);

    dim3 blockDim(WARP_SIZE, BLOCK_COL_WARP_NUM_CHANGE, BLOCK_ROW_WARP_NUM_CHANGE);
    cudaFuncSetAttribute(
        lb_turing_indirect_conv_wmma_int8_ldg16_k64_spilt3x3_relu<
            WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
            WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE,
            OVERLAP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    // cudaMemsetAsync(tmp, 0, GEMM_M * GEMM_N * sizeof(int), CUDA_WORKSPACE_STREAM(ws));
    // cudaMemset(tmp, 0, GEMM_M * GEMM_N * sizeof(int));
    init_zero<<<GEMM_M, GEMM_N / 4, 0, CUDA_WORKSPACE_STREAM(ws)>>>(tmp);
    lb_turing_indirect_conv_wmma_int8_ldg16_k64_spilt3x3_relu<
        WARP_ROW_WMMA_SZ_CHANGE, WARP_COL_WMMA_SZ_CHANGE, WMMA_K_STEP_CHANGE, WMMA_16x16_M,
        WMMA_16x16_N, WMMA_16x16_K, BLOCK_ROW_WARP_NUM_CHANGE, BLOCK_COL_WARP_NUM_CHANGE, OVERLAP>
        <<<gridDim, blockDim, sharedMM_AB_sz, CUDA_WORKSPACE_STREAM(ws)>>>(
            input_tensor, weight_tensor, tmp, GEMM_M, GEMM_N, GEMM_K, GEMM_M_PAD, GEMM_N_PAD,
            GEMM_K_PAD, bias, bias_flag, relu_flag, i_b, i_c, i_h, i_w, k_n, k_c, k_h, k_w, pad_h,
            pad_w, str_h, str_w, out_batch, out_channel, out_height, out_width, bits, alpha,
            preComputeA_ptr);

    quantize<<<GEMM_M, out_channel, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        tmp, output_tensor, bias, alpha, k_n, bias_flag, relu_flag, bits);
}

#endif /*TURING_INDIRECT_CONV_WMMA_LDG16_K64_CUH*/
