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
#include "art/op_tp.h"

typedef struct {
    op_t o;
    int32_t p;
    int32_t axis;
} op_lpnormalization_t;

op_lpnormalization_t *op_default_lpnormalization_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_lpnormalization_t *res = (op_lpnormalization_t *)malloc(sizeof(op_lpnormalization_t));
    memset(res, 0, sizeof(op_lpnormalization_t));
    return res;
}

void op_default_lpnormalization_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_LPNORMALIZATION_P, dtINT32, &((op_lpnormalization_t *)op)->p));
    CHECK(op_setting_single_get(
        op, SETTING_LPNORMALIZATION_AXIS, dtINT32, &((op_lpnormalization_t *)op)->axis));
}

void op_default_lpnormalization_tp_destroy(op_t *op) { (void)op; }

void op_default_lpnormalization_tp_dealloc(op_t *op)
{
    if (NULL != op) {
        free(op);
    }
}

static void op_default_l2norm_run(op_t *op)
{
    op_lpnormalization_t *lpnorm_op = (op_lpnormalization_t *)op;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    int32_t axis = lpnorm_op->axis;
    axis = axis > 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    shape_t *shape = &op->input_tensors[0]->shape;
    size_t reduced_num = shape->dim[axis];
    size_t inner_num = shape_part_count(shape, axis) / reduced_num;
    size_t outer_num = shape_count(shape) / reduced_num / inner_num;

    size_t i, j, k;
    float sum = 0;
    for (i = 0; i < outer_num; i++) {
        for (j = 0; j < inner_num; j++) {
            const float *tmp = &input_0[i * reduced_num * inner_num + j];
            sum = 0.;
            for (k = 0; k < reduced_num; k++) {
                sum += tmp[k * inner_num] * tmp[k * inner_num];
            }
            sum = sqrt(sum);
            for (k = 0; k < reduced_num; k++) {
                output_0[i * reduced_num * inner_num + k * inner_num + j]
                    = tmp[k * inner_num] / sum;
            }
        }
    }
}

static void op_default_l1norm_run(op_t *op)
{
    op_lpnormalization_t *lpnorm_op = (op_lpnormalization_t *)op;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    int32_t axis = lpnorm_op->axis;
    axis = axis > 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    shape_t *shape = &op->input_tensors[0]->shape;
    size_t reduced_num = shape->dim[axis];
    size_t inner_num = shape_part_count(shape, axis) / reduced_num;
    size_t outer_num = shape_count(shape) / reduced_num / inner_num;

    size_t i, j, k;
    float sum = 0;
    for (i = 0; i < outer_num; i++) {
        for (j = 0; j < inner_num; j++) {
            const float *tmp = &input_0[i * reduced_num * inner_num + j];
            sum = 0.;
            for (k = 0; k < reduced_num; k++) {
                sum += fabs(tmp[k * inner_num]);
            }
            for (k = 0; k < reduced_num; k++) {
                output_0[i * reduced_num * inner_num + k * inner_num + j]
                    = tmp[k * inner_num] / sum;
            }
        }
    }
}

void op_default_lpnormalization_tp_prepare(op_t *op)
{
    op_lpnormalization_t *lpnorm_op = (op_lpnormalization_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (1 == lpnorm_op->p) {
            op->run_func = op_default_l1norm_run;
        } else if (2 == lpnorm_op->p) {
            op->run_func = op_default_l2norm_run;
        } else {
            CHECK(false);
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
