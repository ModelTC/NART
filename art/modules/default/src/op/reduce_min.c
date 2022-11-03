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
    bool keepdims;
    int32_t *axes;
    mem_t *reduce;
} op_reducemin_t;

op_reducemin_t *op_default_reducemin_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_reducemin_t *res = (op_reducemin_t *)malloc(sizeof(op_reducemin_t));
    memset(res, 0, sizeof(op_reducemin_t));
    return res;
}

void op_default_reducemin_tp_config(op_t *op)
{
    size_t len;
    CHECK(op_setting_array_get(
        op, SETTING_REDUCE_AXES, dtINT32, &len, &((op_reducemin_t *)op)->axes));
    (void)op;
}

void op_default_reducemin_tp_destroy(op_t *op)
{
    if (((op_reducemin_t *)op)->reduce != NULL) {
        mem_delete(((op_reducemin_t *)op)->reduce);
        ((op_reducemin_t *)op)->reduce = NULL;
    }
    (void)op;
}

void op_default_reducemin_tp_dealloc(op_t *op)
{
    if (NULL != op) {
        free(op);
    }
}

static void reduce_min(const float *input_0, float *output_0, int cnt_i)
{
    int i;
    for (i = 0; i < cnt_i; ++i) {
        output_0[i] = fmin(output_0[i], input_0[i]);
    }
}

static void op_default_reducemin_run(op_t *op)
{
    size_t count = shape_count(&op->input_tensors[0]->shape);
    shape_t shape_i = op->input_tensors[0]->shape;
    size_t count_o = shape_count(&op->output_tensors[0].shape);
    int i, j, k;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    size_t len;
    int *axes;
    CHECK(op_setting_array_get(op, SETTING_REDUCE_AXES, dtINT32, &len, &axes));
    if (len == 0) {
        output_0[0] = FLT_MAX;
        for (i = 0; i < (int)count; ++i) {
            output_0[0] = fmin(output_0[0], input_0[i]);
        }
        return;
    }
    int32_t flag[MAX_DIM] = { 0 };
    size_t count1 = count;
    size_t count2 = 1;
    for (i = 0; i < (int)len; ++i) {
        if (axes[i] < 0)
            axes[i] = shape_i.dim_size + axes[i];
    }
    for (i = 0; i < (int)len; ++i) {
        flag[axes[i]] = 1;
    }
    float *reduce = (float *)mem_data(((op_reducemin_t *)op)->reduce);
    memcpy(reduce, input_0, sizeof(float) * count);
    for (i = 0; i < (int)shape_i.dim_size; ++i) {
        count1 /= shape_i.dim[i];
        if (flag[i]) {
            float *in = reduce;
            float *out = reduce;
            for (j = 0; j < (int)count2; ++j) {
                for (k = 0; k < (int)count1; ++k) {
                    out[k] = in[k];
                }
                in += count1;
                for (k = 1; k < (int)shape_i.dim[i]; ++k) {
                    reduce_min(in, out, count1);
                    in += count1;
                }
                out += count1;
            }
        }
        count2 *= shape_i.dim[i];
    }
    memcpy(output_0, reduce, sizeof(float) * count_o);
}

void op_default_reducemin_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    size_t count = shape_count(&op->input_tensors[0]->shape);
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (((op_reducemin_t *)op)->reduce == NULL) {
            ((op_reducemin_t *)op)->reduce = mem_new(cpu_mem_tp);
        }
        mem_alloc(((op_reducemin_t *)op)->reduce, count * sizeof(float));
        op->run_func = op_default_reducemin_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
