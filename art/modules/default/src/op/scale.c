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

typedef struct {
    op_t o;
    bool bias_term;
} op_scale_t;

op_scale_t *op_default_scale_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_scale_t *res = (op_scale_t *)malloc(sizeof(op_scale_t));
    memset(res, 0, sizeof(op_scale_t));
    return res;
}

void op_default_scale_tp_config(op_t *op)
{
    (void)op;
    CHECK(
        op_setting_single_get(op, SETTING_SCALE_BIAS_TERM, dtBOOL, &((op_scale_t *)op)->bias_term));
}

void op_default_scale_tp_destroy(op_t *op) { (void)op; }

void op_default_scale_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_scale_run_with_bias(op_t *op)
{
    int i, j, k, idx;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    int channel = op->input_tensors[0]->shape.dim[1];
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }
    int chw = channel * hw;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const float *input_2 = mem_cpu_data(op->input_tensors[2]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    // scale shift mean variance
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            for (k = 0; k < hw; ++k) {
                output_0[i * chw + j * hw + k]
                    = input_0[i * chw + j * hw + k] * input_1[j] + input_2[j];
            }
        }
    }
}

static void op_default_scale_run(op_t *op)
{
    int i, j, k, idx;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    int channel = op->input_tensors[0]->shape.dim[1];
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }
    int chw = channel * hw;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    // scale shift mean variance
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            for (k = 0; k < hw; ++k) {
                output_0[i * chw + j * hw + k] = input_0[i * chw + j * hw + k] * input_1[j];
            }
        }
    }
}

void op_default_scale_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (3 == op->input_size) {
            CHECK_EQ(true, ((op_scale_t *)op)->bias_term);
            op->run_func = op_default_scale_run_with_bias;
        } else if (2 == op->input_size) {
            if (true == ((op_scale_t *)op)->bias_term) {
                LOG(warn,
                    "op bias_term == true, but input_size != 3. Continue without bias_term.\n");
            }
            op->run_func = op_default_scale_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
