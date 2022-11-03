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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/tensor.h"

typedef struct {
    op_t o;
    int32_t axis;
    int32_t k;
    mem_t *buffer;
} op_topk_t;

op_topk_t *op_default_topk_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_topk_t *res = (op_topk_t *)malloc(sizeof(op_topk_t));
    memset(res, 0, sizeof(op_topk_t));
    return res;
}

void op_default_topk_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_TOPK_AXIS, dtINT32, &((op_topk_t *)op)->axis));
    CHECK(op_setting_single_get(op, SETTING_TOPK_K, dtINT32, &((op_topk_t *)op)->k));
}

void op_default_topk_tp_destroy(op_t *op)
{
    op_topk_t *topk_op = (op_topk_t *)op;
    if (NULL != topk_op->buffer) {
        mem_delete(topk_op->buffer);
        topk_op->buffer = NULL;
    }
}

void op_default_topk_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void heap_adj(float *val_arr, float *idx_arr, int i, int len)
{
    int n_child;
    float temp_val, temp_idx;
    for (; 2 * i + 1 < len; i = n_child) {
        n_child = 2 * i + 1;
        if (n_child < len - 1 && val_arr[n_child + 1] < val_arr[n_child]) {
            n_child += 1;
        }
        if (val_arr[i] > val_arr[n_child]) {
            temp_val = val_arr[i];
            val_arr[i] = val_arr[n_child];
            val_arr[n_child] = temp_val;

            temp_idx = idx_arr[i];
            idx_arr[i] = idx_arr[n_child];
            idx_arr[n_child] = temp_idx;
        } else {
            break;
        }
    }
}

static void find_topk(
    const float *src, size_t src_pitch, float *values, float *indices, size_t dst_pitch,
    mem_t *mem_buffer, size_t len, int k)
{
    int src_idx, i;
    float temp_val, temp_idx;
    mem_alloc(mem_buffer, len * sizeof(float) * 2);
    float *val_buffer = (float *)mem_data(mem_buffer);
    float *idx_buffer = (float *)mem_data(mem_buffer) + len;
    for (src_idx = 0; src_idx < k; ++src_idx) {
        val_buffer[src_idx] = src[src_idx * src_pitch];
        idx_buffer[src_idx] = src_idx;
    }

    for (i = k / 2 - 1; i >= 0; --i) {
        heap_adj(val_buffer, idx_buffer, i, k);
    }

    for (; src_idx < len; ++src_idx) {
        if (src[src_idx * src_pitch] > val_buffer[0]) {
            val_buffer[0] = src[src_idx * src_pitch];
            idx_buffer[0] = src_idx;
            heap_adj(val_buffer, idx_buffer, 0, k);
        }
    }

    for (i = k - 1; i > 0; --i) {
        temp_val = val_buffer[0];
        val_buffer[0] = val_buffer[i];
        val_buffer[i] = temp_val;

        temp_idx = idx_buffer[0];
        idx_buffer[0] = idx_buffer[i];
        idx_buffer[i] = temp_idx;
        heap_adj(val_buffer, idx_buffer, 0, i);
    }

    for (i = 0; i < k; ++i) {
        values[i * dst_pitch] = val_buffer[i];
        indices[i * dst_pitch] = idx_buffer[i];
    }
}

static void op_default_topk_run(op_t *op)
{
    const op_topk_t *topk = (op_topk_t *)op;
    int32_t axis = topk->axis;
    axis = axis >= 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    int32_t k = topk->k;

    size_t count = shape_count(&op->input_tensors[0]->shape);
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_num = shape_part_count(&op->input_tensors[0]->shape, axis) / ch;
    size_t outer_num = count / inner_num / ch;

    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *values = mem_cpu_data(op->output_tensors[0].mem);
    float *indices = mem_cpu_data(op->output_tensors[1].mem);

    int i, j;
    for (i = 0; i < outer_num; ++i) {
        for (j = 0; j < inner_num; ++j) {
            find_topk(
                &input_0[i * ch * inner_num + j], inner_num, &values[i * k * inner_num + j],
                &indices[i * k * inner_num + j], inner_num, topk->buffer, ch, k);
        }
    }
}

void op_default_topk_tp_prepare(op_t *op)
{
    op_topk_t *topk_op = (op_topk_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    if (NULL == topk_op->buffer) {
        topk_op->buffer = mem_new(cpu_mem_tp);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_topk_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
