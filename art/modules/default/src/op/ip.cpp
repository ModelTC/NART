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
    int32_t axis;
    mem_t *aux_mem_1;
    mem_t *aux_mem_2;
} op_ip_t;
op_ip_t *op_default_ip_tp_alloc(workspace_t *ws);
void op_default_ip_tp_config(op_t *op);
void op_default_ip_tp_destroy(op_t *op);
void op_default_ip_tp_dealloc(op_t *op);
void op_default_ip_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_ip_t *op_default_ip_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_ip_t *res = (op_ip_t *)malloc(sizeof(op_ip_t));
    memset(res, 0, sizeof(op_ip_t));
    return res;
}

void op_default_ip_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_IP_NUM_OUTPUT, dtUINT32, &((op_ip_t *)op)->num_output));
    CHECK(op_setting_single_get(op, SETTING_IP_AXIS, dtINT32, &((op_ip_t *)op)->axis));
}

void op_default_ip_tp_destroy(op_t *op)
{
    mem_delete(((op_ip_t *)op)->aux_mem_1);
    mem_delete(((op_ip_t *)op)->aux_mem_2);
}

void op_default_ip_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_ip_run(op_t *op)
{
    size_t i, axis;
    const op_ip_t *ip_op = (op_ip_t *)op;
    size_t i_preceding_axes = 1;
    size_t i_succeeding_axes = 1;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(float) * shape_count(&op->output_tensors[0].shape));

    if (ip_op->axis < 0) {
        axis = ip_op->axis + op->input_tensors[0]->shape.dim_size;
    } else {
        axis = ip_op->axis;
    }
    for (i = 0; i < axis; ++i) {
        i_preceding_axes *= op->input_tensors[0]->shape.dim[i];
    }
    for (i = axis; i < (size_t)op->input_tensors[0]->shape.dim_size; ++i) {
        i_succeeding_axes *= op->input_tensors[0]->shape.dim[i];
    }

    sgemm_AxB(i_preceding_axes, ip_op->num_output, i_succeeding_axes, input_0, input_1, output_0);
}

static void op_default_ip_bias_run(op_t *op)
{
    size_t i, axis;
    const op_ip_t *ip_op = (op_ip_t *)op;
    size_t i_preceding_axes = 1;
    size_t i_succeeding_axes = 1;
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    const float *input_2 = (const float *)mem_cpu_data(op->input_tensors[2]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);
    memset(output_0, 0, sizeof(float) * shape_count(&op->output_tensors[0].shape));
    float *output_0_temp = output_0;

    if (ip_op->axis < 0) {
        axis = ip_op->axis + op->input_tensors[0]->shape.dim_size;
    } else {
        axis = ip_op->axis;
    }
    for (i = 0; i < axis; ++i) {
        i_preceding_axes *= op->input_tensors[0]->shape.dim[i];
    }
    for (i = axis; i < (size_t)op->input_tensors[0]->shape.dim_size; ++i) {
        i_succeeding_axes *= op->input_tensors[0]->shape.dim[i];
    }

    // prepare bias data
    for (i = 0; i < i_preceding_axes; ++i) {
        memcpy(output_0_temp, input_2, sizeof(float) * ip_op->num_output);
        output_0_temp += ip_op->num_output;
    }

    sgemm_AxB(i_preceding_axes, ip_op->num_output, i_succeeding_axes, input_0, input_1, output_0);
}

void op_default_ip_tp_prepare(op_t *op)
{
    int i;
    op_ip_t *ip_op = (op_ip_t *)op;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    if (NULL == ip_op->aux_mem_1)
        ip_op->aux_mem_1 = mem_new(cpu_mem_tp);
    mem_alloc(ip_op->aux_mem_1, sizeof(float) * shape_count(&op->input_tensors[1]->shape));
    if (NULL == ip_op->aux_mem_2)
        ip_op->aux_mem_2 = mem_new(cpu_mem_tp);
    mem_alloc(ip_op->aux_mem_2, sizeof(float) * shape_count(&op->output_tensors[0].shape));

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (1) {
            if (op->input_size == 2) {
                op->run_func = op_default_ip_run;
            } else {
                op->run_func = op_default_ip_bias_run;
            }
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
