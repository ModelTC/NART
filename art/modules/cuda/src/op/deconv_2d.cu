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

#include <cstdio>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#define TILE_WIDTH             32
#define FETCH_FLOAT4(pointer)  (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_UINT8_4(pointer) (reinterpret_cast<uchar4 *>(&(pointer))[0])
#define FETCH_INT8_4(pointer)  (reinterpret_cast<char4 *>(&(pointer))[0])
#define FETCH_INT32_4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])
#define OFFSET(row, col, ld)   ((row) * (ld) + (col))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t group;

    size_t bottom_offset;
    size_t weight_offset;
    size_t bias_offset;
    size_t top_offset;
    size_t weight_size;
    size_t unroll_h;
    size_t unroll_w;
    bool processed = false;

    mem_t *ws;
    mem_t *input_unroll;
    mem_t *weight_permuted;
    mem_t *output_permuted;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t convIn;
    cudnnFilterDescriptor_t convW;
    cudnnTensorDescriptor_t convBias;
    cudnnTensorDescriptor_t convOut;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;
    cublasHandle_t cublasHandle;
} op_deconv_2d_t;

op_deconv_2d_t *op_cuda_deconv_2d_tp_alloc(workspace_t *ws);
void op_cuda_deconv_2d_tp_config(op_t *op);
void op_cuda_deconv_2d_tp_destroy(op_t *op);
void op_cuda_deconv_2d_tp_dealloc(op_t *op);
void op_cuda_deconv_2d_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif

op_deconv_2d_t *op_cuda_deconv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_deconv_2d_t *res = new op_deconv_2d_t;
    memset(res, 0, sizeof(op_t));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&res->convDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convIn));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&res->convW));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convBias));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convOut));
    res->ws = mem_new(cuda_mem_tp);
    res->input_unroll = mem_new(cuda_mem_tp);
    res->weight_permuted = mem_new(cuda_mem_tp);
    res->output_permuted = mem_new(cuda_mem_tp);
    res->cublasHandle = ((cuda_workspace_t *)ws)->cublasHandle;
    return res;
}

void op_cuda_deconv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_deconv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_deconv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_deconv_2d_t *)op)->kernel_w));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_deconv_2d_t *)op)->pad_h));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_deconv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_deconv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_deconv_2d_t *)op)->stride_w));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_deconv_2d_t *)op)->group));
}

void op_cuda_deconv_2d_tp_destroy(op_t *op) { (void)op; }

void op_cuda_deconv_2d_tp_dealloc(op_t *op)
{
    op_deconv_2d_t *conv_op = (op_deconv_2d_t *)op;
    if (NULL != conv_op->ws) {
        mem_delete(conv_op->ws);
    }
    if (NULL != conv_op->input_unroll) {
        mem_delete(conv_op->input_unroll);
    }
    if (NULL != conv_op->weight_permuted) {
        mem_delete(conv_op->weight_permuted);
    }
    if (NULL != conv_op->output_permuted) {
        mem_delete(conv_op->output_permuted);
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convIn));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(conv_op->convW));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convBias));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convOut));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_op->convDesc));
    delete conv_op;
}

static void op_cuda_deconv_2d_run(op_t *op)
{
    op_deconv_2d_t *conv_op = (op_deconv_2d_t *)op;
    float alpha = 1.0;
    float beta = 0.0;

    uint32_t g;
    for (g = 0; g < conv_op->group; ++g) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), &alpha, conv_op->convW,
            (float *)mem_data(op->input_tensors[1]->mem) + conv_op->weight_offset * g,
            conv_op->convIn,
            (float *)mem_data(op->input_tensors[0]->mem) + conv_op->bottom_offset * g,
            conv_op->convDesc, conv_op->perfResults.algo, mem_data(conv_op->ws),
            mem_sizeof(conv_op->ws), &beta, conv_op->convOut,
            (float *)mem_data(op->output_tensors[0].mem) + conv_op->top_offset * g));
        if (op->input_size > 2) {
            CUDNN_CHECK(cudnnAddTensor(
                CUDA_WORKSPACE_CUDNNHDL(op->workspace), &alpha, conv_op->convBias,
                (float *)mem_data(op->input_tensors[2]->mem) + conv_op->bias_offset * g, &alpha,
                conv_op->convOut,
                (float *)mem_data(op->output_tensors[0].mem) + conv_op->top_offset * g));
        }
    }
}

__host__ __device__ unsigned int
get_fmap_index(unsigned int d0, unsigned int d1, unsigned int d2, unsigned int d3, shape_t shape)
{
    // Input shape: C*W*W
    return d0 * shape.dim[1] * shape.dim[2] * shape.dim[3] + d1 * shape.dim[2] * shape.dim[3]
        + d2 * shape.dim[3] + d3;
}

template <typename T, const int COL_LEN, const int ROW_LEN>
__global__ void unroll_kernel_transpose(
    T *input_ptr, T *input_unroll_ptr, shape_t input_shape, shape_t output_shape, int32_t stride_h,
    int32_t stride_w, int32_t kernel_h, int32_t kernel_w, int32_t pad_h, int32_t pad_w,
    const int h_len, const int w_len, const int shared_inner_row_size, const int shared_x,
    const int shared_y)
{
    int p, q, r, s, tid, shared_h, shared_w, shared_c;
    int o_h = output_shape.dim[2];
    int b = blockIdx.y / o_h;
    int h = blockIdx.y - b * o_h;
    int w_outer = blockIdx.x * shared_inner_row_size;
    int i_c = input_shape.dim[1];
    int i_h = input_shape.dim[2];
    int i_w = input_shape.dim[3];
    int h_start_raw = h - (kernel_h - pad_h - 1);
    int h_start = max(h_start_raw, 0);
    int w_start_outer = max(w_outer - (kernel_w - pad_w - 1), 0);
    int w0_i = (h_start + stride_h - 1) / stride_h;
    int w1_i = (w_start_outer + stride_w - 1) / stride_w;
    __shared__ T shared_input[ROW_LEN][COL_LEN + 1];
    tid = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
    for (p = tid; p < shared_x * shared_y; p += shared_x * shared_inner_row_size) {
        r = p / shared_y;
        shared_c = min(r, i_c - 1);
        q = p - r * shared_y;
        s = q / w_len;
        shared_h = min(s, i_h - w0_i - 1);
        shared_w = min(q - s * w_len, i_w - w1_i - 1);
        shared_input[shared_h * w_len + shared_w][shared_c]
            = input_ptr[get_fmap_index(b, shared_c, w0_i + shared_h, w1_i + shared_w, input_shape)];
    }
    int o_w = output_shape.dim[3];
    int inserted_i_h = i_h * stride_h - (stride_h - 1);
    int inserted_i_w = i_w * stride_w - (stride_w - 1);
    int H_unroll = i_c * kernel_h * kernel_w;
    int inner_row_num = min(shared_inner_row_size, o_w - w_outer);
    int num_elem_of_block = H_unroll * inner_row_num;
    int W_unroll = o_h * o_w;
    int w_unroll = h * o_w + w_outer + W_unroll * b;
    int index_base = H_unroll * w_unroll;
#pragma unroll
    for (p = tid; p < num_elem_of_block; p += shared_x * shared_inner_row_size) {
        input_unroll_ptr[index_base + p] = 0;
    }
    __syncthreads();
    if (threadIdx.y < inner_row_num && threadIdx.x < i_c) {
        index_base += threadIdx.y * H_unroll;
        int w_start_raw = w_outer + threadIdx.y - (kernel_w - pad_w - 1);
        int h_offset = w0_i * stride_h - h_start;
        int w_start = max(0, w_start_raw);
        int w_idx = (w_start + stride_w - 1) / stride_w - w1_i;
        int w_offset = (w_idx + w1_i) * stride_w - w_start;
        int w_idx_orig = w_idx;
        int h_idx = 0;
        for (p = h_start - h_start_raw + h_offset; p < min(kernel_h, inserted_i_h - h_start_raw);
             p += stride_h) {
            for (q = w_start - w_start_raw + w_offset;
                 q < min(kernel_w, inserted_i_w - w_start_raw); q += stride_w) {
                r = threadIdx.x * kernel_h * kernel_w + p * kernel_h + q;
                input_unroll_ptr[index_base + r] = shared_input[w_idx + h_idx * w_len][threadIdx.x];
                w_idx++;
            }
            h_idx++;
            w_idx = w_idx_orig;
        }
    }
}

template <typename T>
__global__ void unroll_kernel_transpose(
    T *input_ptr, T *input_unroll_ptr, shape_t input_shape, shape_t output_shape, uint32_t stride_h,
    uint32_t stride_w, uint32_t kernel_h, uint32_t kernel_w, uint32_t pad_h, uint32_t pad_w)
{
    int i_b = input_shape.dim[0];
    int i_c = input_shape.dim[1];
    int i_h = input_shape.dim[2];
    int i_w = input_shape.dim[3];
    int o_h = output_shape.dim[2];
    int o_w = output_shape.dim[3];
    int inserted_i_h = i_h * stride_h - (stride_h - 1);
    int inserted_i_w = i_w * stride_w - (stride_w - 1);

    int b, c, s, h, w, h_unroll, w_unroll, h_base, p, q;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = o_h * o_w;
    uint32_t N = i_b * i_c * W_unroll;
    int s32_kernel_h = kernel_h;
    int s32_kernel_w = kernel_w;

    if (t < N) {
        b = t / (i_c * W_unroll);
        c = t % (i_c * W_unroll) / W_unroll; // if t < 28*28, c = 0  // output channel
        s = t % W_unroll; // output height * output width
        h = s / o_w; // output height
        w = s % o_w; // output width
        w_unroll = h * o_w + w + W_unroll * b; // in conv1, max 28*28(s)
        h_base = c * kernel_h * kernel_w;

        int h_lower = kernel_h - pad_h - 1 - h;
        int w_lower = kernel_w - pad_w - 1 - w;
        int h_offset
            = (stride_h - (max(h_lower, 0) + h - (kernel_h - pad_h - 1)) % stride_h) % stride_h;
        int w_offset
            = (stride_w - (max(w_lower, 0) + w - (kernel_w - pad_w - 1)) % stride_w) % stride_w;
        int w0_i = (max(h_lower, 0) + h_offset + h - (kernel_h - pad_h - 1)) / stride_h;
        int w1_i = (max(w_lower, 0) + w_offset + w - (kernel_w - pad_w - 1)) / stride_w;
        int orig_w1_i = w1_i;

#pragma unroll
        for (p = 0; p < kernel_h; p++) {
#pragma unroll
            for (q = 0; q < kernel_w; q++) {
                h_unroll = h_base + p * kernel_w + q;
                input_unroll_ptr[h_unroll * (W_unroll * i_b) + w_unroll] = 0;
            }
        }
        for (p = max(h_lower, 0) + h_offset; p < min(s32_kernel_h, inserted_i_h + h_lower);
             p += stride_h) {
            for (q = max(w_lower, 0) + w_offset; q < min(s32_kernel_w, inserted_i_w + w_lower);
                 q += stride_w) {
                h_unroll = h_base + p * kernel_w + q;
                input_unroll_ptr[h_unroll * (W_unroll * i_b) + w_unroll]
                    = input_ptr[get_fmap_index(b, c, w0_i, w1_i, input_shape)];
                w1_i++;
            }
            w0_i++;
            w1_i = orig_w1_i;
        }
    }
}

template <class T_input, class T_filt, class T_out>
__global__ void gemm_h(
    int batch, T_filt *__restrict__ Md, T_input *__restrict__ Nd, T_out *__restrict__ Pd,
    int M_height_in, int M_width_N_height_in, int N_width_in, int height_out, int width_out)
{
    // M x N
    __shared__ T_filt Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ T_input Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T_out Pvalue = 0;
// width
#pragma unroll
    for (int m = 0; m < ceilf((float)M_width_N_height_in / TILE_WIDTH); ++m) {
        if (row < M_height_in && (m * TILE_WIDTH + tx) < M_width_N_height_in) // X
            Mds[ty][tx] = Md[row * M_width_N_height_in + (m * TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0;
        if ((m * TILE_WIDTH + ty) < M_width_N_height_in && col < N_width_in) // W
            Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * N_width_in + col];
        else
            Nds[ty][tx] = 0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }
    // rearange the output sequence of different batch according to the [b, c, h, w]
    int c_height_out = height_out / batch;
    int c_width_out = width_out * batch;
    if (row < c_height_out && col < c_width_out) {
        if (col < width_out) {
            Pd[row * width_out + col] = Pvalue; // Output
        } else {
            int offset = col / width_out;
            Pd[(row + offset * c_height_out) * width_out + col % width_out] = Pvalue;
        }
    }
}

template <
    const int BLOCK_SIZE_M, // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K, // height of block of A that each thread block load into shared
    // memory
    const int BLOCK_SIZE_N, // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X // width of block of C that each thread calculate
    >
__global__ void gemm_h_bias(
    int batch, int8_t *__restrict__ Md, uint8_t *__restrict__ Nd, int32_t *__restrict__ Pd,
    int32_t *__restrict__ B, int M_height_in, int M_width_N_height_in, int N_width_in,
    int height_out, int width_out)
{
    // M x N
    __shared__ int8_t Mds[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ uint8_t Nds[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // registers for C
    int32_t accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0 };
    // registers for A and B
    int8_t frag_a[THREAD_SIZE_Y];
    uint8_t frag_b[THREAD_SIZE_X];

    int8_t int8_zeros[4] = { 0 };
    uint8_t uint8_zeros[4] = { 0 };

    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int M_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int N_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int M_TILE_ROW_START = tid / M_TILE_THREAD_PER_ROW;
    const int N_TILE_ROW_START = tid / N_TILE_THREAD_PER_ROW;

    const int M_TILE_COL = tid % M_TILE_THREAD_PER_ROW * 4;
    const int N_TILE_COL = tid % N_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int M_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / M_TILE_THREAD_PER_ROW;
    const int N_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / N_TILE_THREAD_PER_ROW;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0; tile_idx < M_width_N_height_in; tile_idx += BLOCK_SIZE_K) {
// load M from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += M_TILE_ROW_STRIDE) {
            int r = BLOCK_SIZE_M * by + M_TILE_ROW_START + i;
            int c = tile_idx + M_TILE_COL;
            if (r < M_height_in && (c + 4) <= M_width_N_height_in) {
                FETCH_INT8_4(Mds[M_TILE_ROW_START + i][M_TILE_COL])
                    = FETCH_INT8_4(Md[OFFSET(r, c, M_width_N_height_in)]);
            } else {
                FETCH_INT8_4(Mds[M_TILE_ROW_START + i][M_TILE_COL]) = FETCH_INT8_4(int8_zeros);
            }
        }

// load N from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += N_TILE_ROW_STRIDE) {
            int r = tile_idx + N_TILE_ROW_START + i;
            int c = N_TILE_COL + BLOCK_SIZE_N * bx;
            if (r < M_width_N_height_in && (c + 4) <= N_width_in) {
                FETCH_UINT8_4(Nds[N_TILE_ROW_START + i][N_TILE_COL])
                    = FETCH_UINT8_4(Nd[OFFSET(r, c, N_width_in)]);
            } else {
                FETCH_UINT8_4(Nds[N_TILE_ROW_START + i][N_TILE_COL]) = FETCH_UINT8_4(uint8_zeros);
            }
        }

        __syncthreads();

// compute c
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
// load A from shared memory to register
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = Mds[ty * THREAD_SIZE_Y + thread_y][k];
            }

// load B from shared memory to register
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_UINT8_4(frag_b[thread_x])
                    = FETCH_UINT8_4(Nds[k][THREAD_SIZE_X * tx + thread_x]);
            }

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
        }
        __syncthreads();
    }

    int c_height_out = height_out / batch;
    int c_width_out = width_out * batch;
// store back to C
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            int row = BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y;
            int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
            if (row < c_height_out && col < c_width_out) {
                if (col < width_out) {
                    accum[thread_y][thread_x] += B[row];
                    accum[thread_y][thread_x + 1] += B[row];
                    accum[thread_y][thread_x + 2] += B[row];
                    accum[thread_y][thread_x + 3] += B[row];
                    FETCH_INT32_4(Pd[OFFSET(row, col, width_out)])
                        = FETCH_INT32_4(accum[thread_y][thread_x]);
                } else {
                    accum[thread_y][thread_x] += B[row];
                    accum[thread_y][thread_x + 1] += B[row];
                    accum[thread_y][thread_x + 2] += B[row];
                    accum[thread_y][thread_x + 3] += B[row];
                    FETCH_INT32_4(Pd[OFFSET(
                        row + col / width_out * c_height_out, col % width_out, width_out)])
                        = FETCH_INT32_4(accum[thread_y][thread_x]);
                }
            }
        }
    }
}

template <typename T>
__global__ void add_bias_permute_kernel(
    T *out, T *permute_out, const T *bias, size_t size, uint32_t c_width_out, uint32_t width_out,
    uint32_t out_channels)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        uint32_t imul4 = i * 4;
        uint32_t c = imul4 / c_width_out;
        permute_out[imul4] += bias[c];
        permute_out[imul4 + 1] += bias[c];
        permute_out[imul4 + 2] += bias[c];
        permute_out[imul4 + 3] += bias[c];
        uint32_t c_col = imul4 - c * c_width_out;
        uint32_t b = c_col / width_out;
        uint32_t col = c_col - b * width_out;
        FETCH_INT32_4(out[OFFSET(b * out_channels + c, col, width_out)])
            = FETCH_INT32_4(permute_out[imul4]);
    }
}

template <typename T>
__global__ void permute_output_kernel(
    T *out, T *permute_out, size_t size, uint32_t c_width_out, uint32_t width_out,
    uint32_t out_channels)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        uint32_t imul4 = i * 4;
        uint32_t c = imul4 / c_width_out;
        uint32_t c_col = imul4 - c * c_width_out;
        uint32_t b = c_col / width_out;
        uint32_t col = c_col - b * width_out;
        FETCH_INT32_4(out[OFFSET(b * out_channels + c, col, width_out)])
            = FETCH_INT32_4(permute_out[imul4]);
    }
}

__global__ void permute_kernel(
    int8_t *weight_permuted_ptr, int8_t *weight_ptr, shape_t weight_shape, size_t weight_size)
{
    const uint32_t old_steps[]
        = { weight_shape.dim[1] * weight_shape.dim[2] * weight_shape.dim[3],
            weight_shape.dim[2] * weight_shape.dim[3], weight_shape.dim[2], 1 };
    const uint32_t new_steps[]
        = { weight_shape.dim[0] * weight_shape.dim[2] * weight_shape.dim[3],
            weight_shape.dim[2] * weight_shape.dim[3], weight_shape.dim[2], 1 };
    const uint32_t permute_order[] = { 1, 0, 2, 3 };

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= weight_size)
        return;
    uint32_t temp_idx = index;
    uint32_t old_idx = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t order = permute_order[i];
        old_idx += (temp_idx / new_steps[i]) * old_steps[order];
        temp_idx %= new_steps[i];
    }
    weight_permuted_ptr[index] = weight_ptr[old_idx];
}

__global__ void flip_share_kernel(int8_t *data, shape_t shape)
{
    __shared__ int8_t tile[8][8];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int m = bx / (shape.dim[0]);
    unsigned int c = bx % (shape.dim[0]);
    tile[ty][tx] = data[get_fmap_index(c, m, shape.dim[2] - ty - 1, shape.dim[3] - tx - 1, shape)];
    __syncthreads();
    data[get_fmap_index(c, m, ty, tx, shape)] = tile[ty][tx];
}

static void op_cuda_deconv_uint_2d_run(op_t *op)
{
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 128;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;

    op_deconv_2d_t *conv_op = (op_deconv_2d_t *)op;

    shape_t input_shape = op->input_tensors[0]->shape;
    shape_t weight_shape = op->input_tensors[1]->shape;
    shape_t output_shape = op->output_tensors[0].shape;

    uint8_t *d_ifmap = (uint8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *d_filter = (int8_t *)mem_data(op->input_tensors[1]->mem);
    int32_t *d_ofmap = (int32_t *)mem_data(op->output_tensors[0].mem);
    int32_t *d_bias = (int32_t *)mem_data(op->input_tensors[2]->mem);

    uint8_t *input_unroll_ptr = (uint8_t *)mem_data(conv_op->input_unroll);
    int8_t *weight_permuted_ptr = (int8_t *)mem_data(conv_op->weight_permuted);

    dim3 grid(weight_shape.dim[0] * weight_shape.dim[1], 1, 1);
    dim3 block(weight_shape.dim[2], weight_shape.dim[3], 1);

    if (!conv_op->processed) {
        // transpose and flip the filter
        permute_kernel<<<
            (conv_op->weight_size + 1024 - 1) / 1024, 1024, 0,
            CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            weight_permuted_ptr, d_filter, weight_shape, conv_op->weight_size);
        // shape_t weight_filp_shape = { .dim = { weight_shape.dim[1], weight_shape.dim[0],
        // weight_shape.dim[2], weight_shape.dim[3] } };
        shape_t weight_filp_shape = weight_shape;
        weight_filp_shape.dim[0] = weight_shape.dim[1];
        weight_filp_shape.dim[1] = weight_shape.dim[0];
        flip_share_kernel<<<grid, block, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            weight_permuted_ptr, weight_filp_shape);
        conv_op->processed = true;
    }

    // unroll input
    int num_blocks
        = (input_shape.dim[0] * input_shape.dim[1] * output_shape.dim[2] * output_shape.dim[3]
           + 1024 - 1)
        / 1024;
    unroll_kernel_transpose<<<num_blocks, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        d_ifmap, input_unroll_ptr, input_shape, output_shape, conv_op->stride_h, conv_op->stride_w,
        conv_op->kernel_h, conv_op->kernel_w, conv_op->pad_h, conv_op->pad_w);

    dim3 threadsPerBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 numBlocks(
        ceil((float)conv_op->unroll_w / BLOCK_SIZE_N),
        ceil((float)output_shape.dim[1] / BLOCK_SIZE_M)); // bx = O_WIDTH, by = O_HEIGHT

    if (d_bias != NULL) {
        gemm_h_bias<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
            <<<numBlocks, threadsPerBlock>>>(
                output_shape.dim[0], weight_permuted_ptr, input_unroll_ptr, d_ofmap, d_bias,
                output_shape.dim[1], conv_op->unroll_h, conv_op->unroll_w,
                output_shape.dim[0] * output_shape.dim[1], conv_op->unroll_w / output_shape.dim[0]);
    } else {
        gemm_h<<<numBlocks, threadsPerBlock>>>(
            output_shape.dim[0], weight_permuted_ptr, input_unroll_ptr, d_ofmap,
            output_shape.dim[1], conv_op->unroll_h, conv_op->unroll_w,
            output_shape.dim[0] * output_shape.dim[1], conv_op->unroll_w / output_shape.dim[0]);
    }
}

#if CUBLAS_VERSION >= 11000
static void op_cuda_deconv_int_2d_run(op_t *op)
{
    op_deconv_2d_t *conv_op = (op_deconv_2d_t *)op;

    shape_t input_shape = op->input_tensors[0]->shape;
    shape_t weight_shape = op->input_tensors[1]->shape;
    shape_t output_shape = op->output_tensors[0].shape;

    int8_t *d_ifmap = (int8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *d_filter = (int8_t *)mem_data(op->input_tensors[1]->mem);
    int32_t *d_ofmap = (int32_t *)mem_data(op->output_tensors[0].mem);
    int32_t *d_bias = (int32_t *)mem_data(op->input_tensors[2]->mem);

    int8_t *input_unroll_ptr = (int8_t *)mem_data(conv_op->input_unroll);
    int8_t *weight_permuted_ptr = (int8_t *)mem_data(conv_op->weight_permuted);
    int32_t *output_permuted_ptr = (int32_t *)mem_data(conv_op->output_permuted);

    dim3 grid(weight_shape.dim[0] * weight_shape.dim[1], 1, 1);
    dim3 block(weight_shape.dim[2], weight_shape.dim[2], 1);

    if (!conv_op->processed) {
        // transpose and flip the filter
        permute_kernel<<<
            (conv_op->weight_size + 1024 - 1) / 1024, 1024, 0,
            CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            weight_permuted_ptr, d_filter, weight_shape, conv_op->weight_size);
        // shape_t weight_filp_shape = { .dim = { weight_shape.dim[1], weight_shape.dim[0],
        // weight_shape.dim[2], weight_shape.dim[3] } };
        shape_t weight_filp_shape = weight_shape;
        weight_filp_shape.dim[0] = weight_shape.dim[1];
        weight_filp_shape.dim[1] = weight_shape.dim[0];
        flip_share_kernel<<<grid, block, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            weight_permuted_ptr, weight_filp_shape);
        conv_op->processed = true;
    }
    // unroll input
    const int THREAD_NUM_PER_BLOCK = 1024;
    const int ROW_LEN = 40;
    const int shared_x = ((input_shape.dim[1] + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    const int shared_inner_row_size = THREAD_NUM_PER_BLOCK / shared_x;
    int h_len = (conv_op->kernel_h + conv_op->stride_h - 1) / conv_op->stride_h;
    int w_len
        = (conv_op->kernel_w + shared_inner_row_size + conv_op->stride_w - 1) / conv_op->stride_w;
    const int shared_y = h_len * w_len;
    block.x = shared_x;
    block.y = shared_inner_row_size;
    grid.x = (output_shape.dim[3] + shared_inner_row_size - 1) / shared_inner_row_size;
    grid.y = input_shape.dim[0] * output_shape.dim[2];
    unroll_kernel_transpose<int8_t, THREAD_NUM_PER_BLOCK, ROW_LEN>
        <<<grid, block, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            d_ifmap, input_unroll_ptr, input_shape, output_shape, conv_op->stride_h,
            conv_op->stride_w, conv_op->kernel_h, conv_op->kernel_w, conv_op->pad_h, conv_op->pad_w,
            h_len, w_len, shared_inner_row_size, shared_x, shared_y);

    int32_t alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasSetStream(
        CUDA_WORKSPACE_CUBLASHDL(op->workspace), CUDA_WORKSPACE_STREAM(op->workspace)));
    cublasGemmEx(
        conv_op->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, conv_op->unroll_w, output_shape.dim[1],
        conv_op->unroll_h, &alpha, input_unroll_ptr, CUDA_R_8I, conv_op->unroll_h,
        weight_permuted_ptr, CUDA_R_8I, conv_op->unroll_h, &beta, output_permuted_ptr, CUDA_R_32I,
        conv_op->unroll_w, CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    size_t count = shape_count(&op->output_tensors[0].shape);
    if (d_bias != NULL) {
        add_bias_permute_kernel<<<
            ((count + 1024 * 4 - 1) / 1024 / 4), 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            d_ofmap, output_permuted_ptr, d_bias, count / 4, conv_op->unroll_w,
            conv_op->unroll_w / output_shape.dim[0], output_shape.dim[1]);
    } else {
        permute_output_kernel<<<
            ((count + 1024 * 4 - 1) / 1024 / 4), 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            d_ofmap, output_permuted_ptr, count / 4, conv_op->unroll_w,
            conv_op->unroll_w / output_shape.dim[0], output_shape.dim[1]);
    }
}
#endif // CUBLAS_VERSION >= 11000

void op_cuda_deconv_2d_tp_prepare(op_t *op)
{
    op_deconv_2d_t *conv_op = (op_deconv_2d_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }

    if (op->input_tensors[0]->dtype == dtFLOAT32) {
        // chaneg output dtype
        op->output_tensors[0].dtype = dtFLOAT32;
        tensor_alloc(&op->output_tensors[0]);

        /* set input tensor desc */
        shape_t *shape = &op->input_tensors[0]->shape;
        conv_op->bottom_offset = shape->dim[1] / conv_op->group * shape->dim[2] * shape->dim[3];
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            conv_op->convIn, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1] / conv_op->group,
            shape->dim[2], shape->dim[3], // shape
            shape->dim[1] * shape->dim[2] * shape->dim[3], shape->dim[2] * shape->dim[3],
            shape->dim[3], 1)); // stride

        /* set filter desc */
        shape = &op->input_tensors[1]->shape;
        conv_op->weight_offset
            = (shape->dim[0] / conv_op->group) * shape->dim[1] * shape->dim[2] * shape->dim[3];
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            conv_op->convW, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, shape->dim[0] / conv_op->group,
            shape->dim[1], shape->dim[2], shape->dim[3]));

        /* set bias desc */
        if (op->input_size > 2) {
            conv_op->bias_offset = op->input_tensors[2]->shape.dim[0] / conv_op->group;
            CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
                conv_op->convBias, CUDNN_DATA_FLOAT, 1,
                op->input_tensors[2]->shape.dim[0] / conv_op->group, 1, 1,
                op->input_tensors[2]->shape.dim[0] / conv_op->group, 1, 1, 1));
        }

        /* set output tensor desc */
        shape = &op->output_tensors[0].shape;
        conv_op->top_offset = shape->dim[1] / conv_op->group * shape->dim[2] * shape->dim[3];
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            conv_op->convOut, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1] / conv_op->group,
            shape->dim[2], shape->dim[3], shape->dim[1] * shape->dim[2] * shape->dim[3],
            shape->dim[2] * shape->dim[3], shape->dim[3], 1));

        /* set conv desc */
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_op->convDesc, conv_op->pad_h, conv_op->pad_w, conv_op->stride_h, conv_op->stride_w,
            1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* find algo */
        int returnedAlgoCount = 0;

#if 1 // CUDNN_VERSION >= 8000
        CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convW, conv_op->convIn,
            conv_op->convDesc, conv_op->convOut, 1, &returnedAlgoCount, &conv_op->perfResults));
#else
        conv_op->perfResults.algo = (cudnnConvolutionBwdDataAlgo_t)0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convW, conv_op->convIn,
            conv_op->convDesc, conv_op->convOut, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            8 * 1024 * 1024, &conv_op->perfResults.algo));
#endif

        /* get workspace size */
        size_t sizeBytes = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convW, conv_op->convIn,
            conv_op->convDesc, conv_op->convOut, conv_op->perfResults.algo, &sizeBytes));
        mem_alloc(conv_op->ws, sizeBytes);

        op->run_func = op_cuda_deconv_2d_run;
    } else if (op->input_tensors[0]->dtype == dtUINT8 || op->input_tensors[0]->dtype == dtINT8) {
        CUBLAS_CHECK(cublasSetStream(
            CUDA_WORKSPACE_CUBLASHDL(op->workspace), CUDA_WORKSPACE_STREAM(op->workspace)));
        tensor_alloc(&op->output_tensors[0]);
        shape_t input_shape = op->input_tensors[0]->shape;
        shape_t weight_shape = op->input_tensors[1]->shape;
        shape_t output_shape = op->output_tensors[0].shape;
        conv_op->unroll_h = input_shape.dim[1] * weight_shape.dim[2] * weight_shape.dim[3];
        conv_op->unroll_w = output_shape.dim[0] * output_shape.dim[2] * output_shape.dim[3];
        conv_op->weight_size
            = weight_shape.dim[0] * weight_shape.dim[1] * weight_shape.dim[2] * weight_shape.dim[3];
        if (conv_op->unroll_h % 4 != 0 || conv_op->unroll_w % 4 != 0) {
            fprintf(
                stderr,
                "input channel multiply kernel's height and weight must be a multiple of 4! ");
            abort();
        }
        mem_alloc(conv_op->weight_permuted, conv_op->weight_size * sizeof(int8_t));
        mem_alloc(conv_op->input_unroll, conv_op->unroll_h * conv_op->unroll_w * sizeof(uint8_t));

        if (op->input_tensors[0]->dtype == dtUINT8) {
            op->run_func = op_cuda_deconv_uint_2d_run;
        } else {
#if CUBLAS_VERSION >= 11000
            if ((output_shape.dim[2] * output_shape.dim[3]) % 4 != 0) {
                fprintf(stderr, "product of output height and width must be a multiple of 4! ");
                abort();
            }
            mem_alloc(
                conv_op->output_permuted,
                weight_shape.dim[1] * conv_op->unroll_w * sizeof(int32_t));
            op->run_func = op_cuda_deconv_int_2d_run;
            if (!conv_op->processed) {
                // Following is to assure shared memory required is strictly less than 48KB
                CHECK(input_shape.dim[1] <= 1024);
                int shared_x = ((input_shape.dim[1] + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
                // Number of output elements that a block deals with
                int shared_inner_row_size = 1024 / shared_x;
                // Number of elements in h dimension of input that affect an element in output
                int h_len = (conv_op->kernel_h + conv_op->stride_h - 1) / conv_op->stride_h;
                // Number of elements in w dimension of input that affect shared_inner_row_size
                // elements in a row in output
                int w_len = (conv_op->kernel_w + shared_inner_row_size + conv_op->stride_w - 1)
                    / conv_op->stride_w;
                CHECK(h_len * w_len <= 40);
            }
#else
            CHECK(false);
#endif // CUBLAS_VERSION >= 11000
        }
    } else {
        CHECK(false);
    }
}
