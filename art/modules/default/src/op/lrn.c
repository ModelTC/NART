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

#include "art/op_tp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"

    typedef struct {
    op_t o;
    uint32_t local_size;
    float alpha;
    float beta;
    float k;
    uint32_t norm_region; // 默认是做通道间的归一化
    mem_t *scale_data;
    mem_t *padded_data;
} op_lrn_t;

op_lrn_t *op_default_lrn_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_lrn_t *res = (op_lrn_t *)malloc(sizeof(op_lrn_t));
    memset(res, 0, sizeof(op_lrn_t));
    return res;
}

void op_default_lrn_tp_config(op_t *op)
{
    CHECK(
        op_setting_single_get(op, SETTING_LRN_LOCAL_SIZE, dtUINT32, &((op_lrn_t *)op)->local_size));
    CHECK(op_setting_single_get(op, SETTING_LRN_ALPHA, dtFLOAT32, &((op_lrn_t *)op)->alpha));
    CHECK(op_setting_single_get(op, SETTING_LRN_BETA, dtFLOAT32, &((op_lrn_t *)op)->beta));
    CHECK(op_setting_single_get(op, SETTING_LRN_K, dtFLOAT32, &((op_lrn_t *)op)->k));
    CHECK(op_setting_single_get(
        op, SETTING_LRN_NORM_REGION, dtUINT32, &((op_lrn_t *)op)->norm_region));
}

void op_default_lrn_tp_destroy(op_t *op)
{
    if (((op_lrn_t *)op)->scale_data)
        mem_delete(((op_lrn_t *)op)->scale_data);
    if (((op_lrn_t *)op)->padded_data)
        mem_delete(((op_lrn_t *)op)->padded_data);
    ((op_lrn_t *)op)->scale_data = NULL;
    ((op_lrn_t *)op)->padded_data = NULL;
}

void op_default_lrn_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_lrn_run(op_t *op)
{
    //这里采用caffe对于lrn的实现
    const op_lrn_t *lrn_op = (op_lrn_t *)op;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    int channel = op->input_tensors[0]->shape.dim[1];
    int hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    int chw = channel * hw;
    int i, j, n;
    int local_size = lrn_op->local_size;
    size_t copy_count;
    float beta = lrn_op->beta;
    float k = lrn_op->k;
    float alpha_div_n = lrn_op->alpha / local_size;
    int padded_offset = local_size / 2 * hw;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    float *padded_data = (float *)mem_cpu_data(lrn_op->padded_data);
    float *padded_data_prev = NULL;
    float *padded_data_temp = NULL;
    float *scale_data = (float *)mem_cpu_data(lrn_op->scale_data);
    padded_data_temp = padded_data;
    copy_count = sizeof(float) * chw;
    for (n = 0; n < batch_size; ++n) {
        padded_data = padded_data_temp;
        memcpy(padded_data + padded_offset, input_0, copy_count);
        padded_data_prev = padded_data;
        for (i = 0; i < local_size; ++i) {
            for (j = 0; j < hw; ++j) {
                scale_data[j] += padded_data[j];
            }
            padded_data += hw;
        }
        for (i = 0; i < hw; ++i) {
            output_0[i] = input_0[i] / pow(k + alpha_div_n * scale_data[i], beta);
        }
        input_0 += hw;
        output_0 += hw;

        for (i = 1; i < channel; ++i) {
            for (j = 0; j < hw; ++j) {
                scale_data[j] -= padded_data[j];
                scale_data[j] += padded_data_prev[j];
                output_0[j] = input_0[j] / pow(k + alpha_div_n * scale_data[j], beta);
            }
            padded_data += hw;
            padded_data_prev += hw;
            input_0 += hw;
            output_0 += hw;
        }
    }
}

void op_default_lrn_tp_prepare(op_t *op)
{
    int i;
    op_lrn_t *lrn_op = (op_lrn_t *)op;
    int hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
    int count = (op->input_tensors[0]->shape.dim[1] + lrn_op->local_size - 1) * hw;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (NULL == lrn_op->scale_data)
            lrn_op->scale_data = mem_new(cpu_mem_tp);
        mem_alloc(lrn_op->scale_data, sizeof(float) * hw);
        if (NULL == lrn_op->padded_data)
            lrn_op->padded_data = mem_new(cpu_mem_tp);
        mem_alloc(lrn_op->scale_data, sizeof(float) * hw);
        mem_alloc(lrn_op->padded_data, sizeof(float) * count);
        memset(mem_cpu_data(lrn_op->scale_data), 0, sizeof(float) * hw);
        memset(mem_cpu_data(lrn_op->padded_data), 0, sizeof(float) * count);
        op->run_func = op_default_lrn_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
