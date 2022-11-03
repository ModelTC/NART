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

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../utils/utils.h"

typedef struct {
    op_t o;
    int32_t axis;
    int32_t keepdims;
    int32_t select_last_index;
    mem_t *max_value;
} op_argmax_t;

op_argmax_t *op_default_argmax_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_argmax_t *res = (op_argmax_t *)malloc(sizeof(op_argmax_t));
    memset(res, 0, sizeof(op_argmax_t));
    return res;
}

void op_default_argmax_tp_config(op_t *op)
{
    op_argmax_t *argmax_op = (op_argmax_t *)op;
    CHECK(op_setting_single_get(op, SETTING_ARGMAX_AXIS, dtINT32, &(argmax_op->axis)));
    CHECK(op_setting_single_get(op, SETTING_ARGMAX_KEEPDIMS, dtINT32, &(argmax_op->keepdims)));
    CHECK(op_setting_single_get(
        op, SETTING_ARGMAX_SELECT_LAST_INDEX, dtINT32, &(argmax_op->select_last_index)));
}

void op_default_argmax_tp_destroy(op_t *op)
{
    op_argmax_t *argmax_op = (op_argmax_t *)op;
    if (argmax_op->max_value != NULL) {
        mem_delete(argmax_op->max_value);
        argmax_op->max_value = NULL;
    }
}

void op_default_argmax_tp_dealloc(op_t *op)
{
    if (NULL != op) {
        free(op);
    }
}

static void op_default_argmax_select_first_run(op_t *op)
{
    op_argmax_t *argmax_op = (op_argmax_t *)op;
    const float *input = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_cpu_data(op->output_tensors[0].mem);
    float *max_val = (float *)mem_cpu_data(argmax_op->max_value);

    size_t i, j, k;
    int axis = argmax_op->axis;
    axis = axis >= 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    size_t axis_cnt = op->input_tensors[0]->shape.dim[axis];
    size_t inner_cnt = shape_part_count(&op->input_tensors[0]->shape, axis) / axis_cnt;
    size_t outer_cnt = shape_count(&op->input_tensors[0]->shape) / inner_cnt / axis_cnt;

    for (i = 0; i < outer_cnt; i++) {
        for (j = 0; j < inner_cnt; j++) {
            max_val[j] = -FLT_MAX;
        }
        for (j = 0; j < axis_cnt; j++) {
            const float *data_ptr = input + i * axis_cnt * inner_cnt + j * inner_cnt;
            float *output_ptr = output + i * inner_cnt;
            for (k = 0; k < inner_cnt; k++) {
                if (data_ptr[k] > max_val[k]) {
                    max_val[k] = data_ptr[k];
                    output_ptr[k] = j;
                }
            }
        }
    }
}

static void op_default_argmax_select_last_run(op_t *op)
{
    op_argmax_t *argmax_op = (op_argmax_t *)op;
    const float *input = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_cpu_data(op->output_tensors[0].mem);
    float *max_val = (float *)mem_cpu_data(argmax_op->max_value);

    size_t i, j, k;
    int axis = argmax_op->axis;
    axis = axis >= 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    size_t axis_cnt = op->input_tensors[0]->shape.dim[axis];
    size_t inner_cnt = shape_part_count(&op->input_tensors[0]->shape, axis) / axis_cnt;
    size_t outer_cnt = shape_count(&op->input_tensors[0]->shape) / inner_cnt / axis_cnt;

    for (i = 0; i < outer_cnt; i++) {
        for (j = 0; j < inner_cnt; j++) {
            max_val[j] = -FLT_MAX;
        }
        for (j = 0; j < axis_cnt; j++) {
            const float *data_ptr = input + i * axis_cnt * inner_cnt + j * inner_cnt;
            float *output_ptr = output + i * inner_cnt;
            for (k = 0; k < inner_cnt; k++) {
                if (data_ptr[k] >= max_val[k]) {
                    max_val[k] = data_ptr[k];
                    output_ptr[k] = j;
                }
            }
        }
    }
}

void op_default_argmax_tp_prepare(op_t *op)
{
    op_argmax_t *argmax_op = (op_argmax_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    int32_t axis = argmax_op->axis >= 0 ? argmax_op->axis
                                        : argmax_op->axis + op->input_tensors[0]->shape.dim_size;
    size_t count = shape_part_count(&op->input_tensors[0]->shape, axis)
        / op->input_tensors[0]->shape.dim[axis];
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (argmax_op->max_value == NULL)
            argmax_op->max_value = mem_new(cpu_mem_tp);
        mem_alloc(argmax_op->max_value, count * sizeof(float));

        if (argmax_op->select_last_index)
            op->run_func = op_default_argmax_select_last_run;
        else
            op->run_func = op_default_argmax_select_first_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
