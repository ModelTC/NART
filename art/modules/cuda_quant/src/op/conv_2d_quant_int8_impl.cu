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

#include "../utils/cuda_quant_helper.cuh"

//#define TIO

#ifdef TIO
#define TI(tag)                           \
    cudaEvent_t _event_start_##tag;       \
    cudaEvent_t _event_end_##tag;         \
    float _event_time_##tag;              \
    cudaEventCreate(&_event_start_##tag); \
    cudaEventCreate(&_event_end_##tag);   \
    cudaEventRecord(_event_start_##tag);

#define TO(tag, str)                                                                \
    cudaEventRecord(_event_end_##tag);                                              \
    cudaEventSynchronize(_event_end_##tag);                                         \
    cudaEventElapsedTime(&_event_time_##tag, _event_start_##tag, _event_end_##tag); \
    printf("%s: %.6fms\n", str, _event_time_##tag);

#else

#define TI(tag)
#define TO(tag, str)

#endif // TIO

void op_cuda_quant_conv_2d_i8xi8_run(op_t *op)
{

    TI(conv_start);
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;

    shape_t *input_shape = &op->input_tensors[0]->shape;
    shape_t *weight_shape = &op->input_tensors[1]->shape;
    shape_t *output_shape = &op->output_tensors[0].shape;
    // todo
    int i_b = input_shape->dim[0];
    int i_c = input_shape->dim[1];
    int i_h = input_shape->dim[2];
    int i_w = input_shape->dim[3];

    int k_n = output_shape->dim[1];
    int k_c = i_c;
    int k_h = weight_shape->dim[2];
    int k_w = weight_shape->dim[3];

    int8_t *input_ptr = (int8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *weight_ptr = (int8_t *)mem_data(op->input_tensors[1]->mem);

    int8_t *output_ptr = (int8_t *)mem_data(op->output_tensors[0].mem);

    size_t output_num = shape_count(output_shape);

    // size_t o_h = output_shape->dim[2];
    // size_t o_w = output_shape->dim[3];

    TO(conv_start, "conv_start");

    TI(indirect_conv);
    bool has_bias = false;
    float *bias_ptr = NULL;
    if (op->input_size > 2) {
        bias_ptr = (float *)mem_data(op->input_tensors[2]->mem);
        has_bias = true;
    }
    // printf("has bias %d\n", has_bias);
    uint8_t out_bit = conv_op->obits[0];

    void *pre_ptr = (void *)mem_data(conv_op->ws_pre);
    float *alphas = (float *)mem_data(conv_op->ws_alphas);
    int *tmp = (int *)mem_data(conv_op->ws_inttmp);
    // printf("conv : relu %d, bias %d\n", (bool)conv_op->relu_flag, has_bias);
    if (conv_op->use_conv_split) {
        conv_op->conv_best_func_split(
            (char *)input_ptr, (char *)weight_ptr, (char *)output_ptr, tmp, i_b, i_c, i_h, i_w, k_n,
            k_c, k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
            (int)conv_op->stride_w, bias_ptr, has_bias, (bool)conv_op->relu_flag, out_bit, alphas,
            pre_ptr, op->workspace);
    } else {
        conv_op->conv_best_func(
            (char *)input_ptr, (char *)weight_ptr, (char *)output_ptr, i_b, i_c, i_h, i_w, k_n, k_c,
            k_h, k_w, (int)conv_op->pad_h, (int)conv_op->pad_w, (int)conv_op->stride_h,
            (int)conv_op->stride_w, bias_ptr, has_bias, (bool)conv_op->relu_flag, out_bit, alphas,
            pre_ptr, op->workspace);
    }
    TO(indirect_conv, "indirect_conv");
}
