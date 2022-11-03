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

#include "../utils/utils.h"

typedef struct {
    op_t o;
    bool share;
} op_prelu_t;

op_prelu_t *op_default_prelu_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_prelu_t *res = (op_prelu_t *)malloc(sizeof(op_prelu_t));
    memset(res, 0, sizeof(op_prelu_t));
    return res;
}

void op_default_prelu_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_PRELU_SHARE, dtBOOL, &((op_prelu_t *)op)->share));
}

void op_default_prelu_tp_destroy(op_t *op) { (void)op; }

void op_default_prelu_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_prelu_run(op_t *op)
{
    size_t n, c, i;
    size_t batch_size = op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.batch_axis];
    size_t channel = op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.channel_axis];
    size_t hw = shape_count(&(op->input_tensors[0]->shape)) / batch_size / channel;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            for (i = 0; i < hw; ++i) {
                *output_0 = MAX(0.f, *input_0) + input_1[c] * MIN(0.f, *input_0);
                ++input_0;
                ++output_0;
            }
        }
    }
}

static void op_default_prelu_share_run(op_t *op)
{
    size_t i;
    size_t count = shape_count(&(op->input_tensors[0]->shape));
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    // const float* input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const float slope = ((float *)mem_cpu_data(op->input_tensors[1]->mem))[0];
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (i = 0; i < count; ++i) {
        *output_0 = MAX(0.f, *input_0) + slope * MIN(0.f, *input_0);
        ++input_0;
        ++output_0;
    }
}
void op_default_prelu_tp_prepare(op_t *op)
{
    int i;
    bool share;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    share = ((op_prelu_t *)op)->share;
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (share) {
            op->run_func = op_default_prelu_share_run;
        } else {
            op->run_func = op_default_prelu_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
