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

#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../utils/im2col.hpp"
#include "../utils/sgemm.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t groups;
    mem_t *temp_data; // temp mem for col in im2col
    mem_t *aux_mem_1; // temp mem for matrix A in sgemm
    mem_t *aux_mem_2; // temp mem for matrix C in sgemm
} op_correlation_t;
op_correlation_t *op_default_correlation_tp_alloc(workspace_t *ws);
void op_default_correlation_tp_config(op_t *op);
void op_default_correlation_tp_destroy(op_t *op);
void op_default_correlation_tp_dealloc(op_t *op);
void op_default_correlation_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_correlation_t *op_default_correlation_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_correlation_t *res = (op_correlation_t *)malloc(sizeof(op_correlation_t));
    memset(res, 0, sizeof(op_correlation_t));
    return res;
}

void op_default_correlation_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION_GROUPS, dtUINT32, &((op_correlation_t *)op)->groups));
}

void op_default_correlation_tp_destroy(op_t *op)
{
    if (((op_correlation_t *)op)->aux_mem_1)
        mem_delete(((op_correlation_t *)op)->aux_mem_1);
    if (((op_correlation_t *)op)->aux_mem_2)
        mem_delete(((op_correlation_t *)op)->aux_mem_2);
    if (((op_correlation_t *)op)->temp_data)
        mem_delete(((op_correlation_t *)op)->temp_data);
}

void op_default_correlation_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_correlation_blas_run(op_t *op)
{
    size_t i, j;
    const op_correlation_t *conv_op = (op_correlation_t *)op;
    size_t kernel_size = conv_op->kernel_h * conv_op->kernel_w * op->input_tensors[0]->shape.dim[1];
    size_t batch_size = op->input_tensors[1]->shape.dim[0];
    size_t num_scales = op->input_tensors[0]->shape.dim[0] / batch_size;
    size_t i_chw = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
        * op->input_tensors[0]->shape.dim[3];
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_chw = o_hw * conv_op->num_output;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(float) * shape_count(&op->output_tensors[0].shape));

    size_t kernel_spatial = shape_count(&op->input_tensors[1]->shape) / batch_size;
    size_t input_spatial = shape_count(&op->input_tensors[0]->shape)
        / op->input_tensors[0]->shape.dim[0] * num_scales;
    for (i = 0; i < batch_size; ++i) {
        input_0 = (float *)mem_cpu_data(op->input_tensors[0]->mem) + i * input_spatial;
        input_1 = (float *)mem_cpu_data(op->input_tensors[1]->mem) + i * kernel_spatial;
        for (j = 0; j < num_scales; ++j) {
            im2col(
                input_0, (float *)mem_cpu_data(conv_op->temp_data),
                op->input_tensors[0]->shape.dim[1], op->input_tensors[0]->shape.dim[2],
                op->input_tensors[0]->shape.dim[3], conv_op->kernel_h, conv_op->kernel_w, 0, 0, 1,
                1, 1, 1);
            // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, conv_op->num_output, o_hw,
            // kernel_size,
            //     1.0, input_1, kernel_size, conv_op->temp_data, kernel_size, 0.0, output_0, o_hw);
            sgemm(
                conv_op->num_output, o_hw, kernel_size, input_1,
                (float *)mem_cpu_data(conv_op->temp_data), output_0, conv_op->aux_mem_1,
                conv_op->aux_mem_2);
            input_0 += i_chw;
            output_0 += o_chw;
        }
    }
}

static void op_default_correlation_blas_group_run(op_t *op)
{
    size_t i, j;
    const op_correlation_t *conv_op = (op_correlation_t *)op;
    size_t kernel_size = conv_op->kernel_h * conv_op->kernel_w;
    size_t batch_size = op->input_tensors[1]->shape.dim[0];
    size_t i_chw = op->input_tensors[0]->shape.dim[1] / conv_op->groups
        * op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    size_t o_hw = op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];
    size_t o_chw = o_hw * conv_op->num_output / conv_op->groups;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(float) * shape_count(&op->output_tensors[0].shape));

    size_t kernel_spatial = shape_count(&op->input_tensors[1]->shape) / batch_size;
    size_t input_spatial
        = shape_count(&op->input_tensors[0]->shape) / op->input_tensors[0]->shape.dim[0];

    for (i = 0; i < batch_size; ++i) {
        input_0 = (float *)mem_cpu_data(op->input_tensors[0]->mem) + i * input_spatial;
        input_1 = (float *)mem_cpu_data(op->input_tensors[1]->mem) + i * kernel_spatial;
        for (j = 0; (int)j < conv_op->groups; ++j) {
            im2col(
                input_0, (float *)mem_cpu_data(conv_op->temp_data),
                op->input_tensors[0]->shape.dim[1] / conv_op->groups,
                op->input_tensors[0]->shape.dim[2], op->input_tensors[0]->shape.dim[3],
                conv_op->kernel_h, conv_op->kernel_w, 0, 0, 1, 1, 1, 1);
            sgemm(
                1, o_hw, kernel_size, input_1, (float *)mem_cpu_data(conv_op->temp_data), output_0,
                conv_op->aux_mem_1, conv_op->aux_mem_2);
            input_0 += i_chw;
            input_1 += kernel_size;
            output_0 += o_chw;
        }
    }
}

void op_default_correlation_tp_prepare(op_t *op)
{
    int i;
    op_correlation_t *conv_op = (op_correlation_t *)op;
    if (conv_op->groups == 1)
        conv_op->num_output
            = op->input_tensors[1]->shape.dim[1] / op->input_tensors[0]->shape.dim[1];
    else
        conv_op->num_output = op->input_tensors[1]->shape.dim[1];
    conv_op->kernel_h = op->input_tensors[1]->shape.dim[2];
    conv_op->kernel_w = op->input_tensors[1]->shape.dim[3];
    size_t count;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    if (NULL == conv_op->aux_mem_1)
        conv_op->aux_mem_1 = mem_new(cpu_mem_tp);
    // mem_alloc(conv_op->aux_mem_1, sizeof(float) * shape_count(&op->input_tensors[1]->shape));
    mem_alloc(
        conv_op->aux_mem_1,
        sizeof(float) * shape_count(&op->input_tensors[1]->shape)
            / op->input_tensors[1]->shape.dim[0]);
    if (NULL == conv_op->aux_mem_2)
        conv_op->aux_mem_2 = mem_new(cpu_mem_tp);
    // mem_alloc(conv_op->aux_mem_2, sizeof(float) * shape_count(&op->output_tensors[0].shape));
    mem_alloc(
        conv_op->aux_mem_2,
        sizeof(float) * op->output_tensors[0].shape.dim[1] * op->output_tensors[0].shape.dim[2]
            * op->output_tensors[0].shape.dim[3]);

    count = conv_op->kernel_h * conv_op->kernel_w * op->input_tensors[0]->shape.dim[1]
        * op->output_tensors[0].shape.dim[2] * op->output_tensors[0].shape.dim[3];

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (NULL == conv_op->temp_data)
            conv_op->temp_data = mem_new(cpu_mem_tp);
        mem_alloc(conv_op->temp_data, sizeof(float) * count);
        memset(mem_cpu_data(conv_op->temp_data), 0, sizeof(float) * count);

        if (conv_op->groups == 1)
            op->run_func = op_default_correlation_blas_run;
        else
            op->run_func = op_default_correlation_blas_group_run;
        break;
    default:
        CHECK(false);
        break;
    }

    if (conv_op->groups != 1) {
        conv_op->groups = op->input_tensors[0]->shape.dim[1];
    }
}
