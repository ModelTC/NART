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
    float eps;
    mem_t *mean;
    mem_t *variance;
} op_batchnorm_t;

op_batchnorm_t *op_default_batchnorm_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_batchnorm_t *res = (op_batchnorm_t *)malloc(sizeof(op_batchnorm_t));
    memset(res, 0, sizeof(op_batchnorm_t));
    return res;
}

void op_default_batchnorm_tp_config(op_t *op)
{
    // todo
    CHECK(
        op_setting_single_get(op, SETTING_BATCHNORM_EPS, dtFLOAT32, &((op_batchnorm_t *)op)->eps));
}

void op_default_batchnorm_tp_destroy(op_t *op)
{
    if (((op_batchnorm_t *)op)->mean != NULL) {
        mem_delete(((op_batchnorm_t *)op)->mean);
        ((op_batchnorm_t *)op)->mean = NULL;
    }
    if (((op_batchnorm_t *)op)->variance != NULL) {
        mem_delete(((op_batchnorm_t *)op)->variance);
        ((op_batchnorm_t *)op)->variance = NULL;
    }
    (void)op;
}

void op_default_batchnorm_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_batchnorm_run(op_t *op)
{
    const op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    int i, j, k, idx;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    int channel = op->input_tensors[0]->shape.dim[1];
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }
    int chw = channel * hw;
    int nhw = batch_size * hw;
    float std;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    // const float* input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    // const float* input_2 = mem_cpu_data(op->input_tensors[2]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    float *mean = (float *)mem_data(((op_batchnorm_t *)op)->mean);
    float *variance = (float *)mem_data(((op_batchnorm_t *)op)->variance);
    memset(mean, 0, sizeof(float) * channel);
    memset(variance, 0, sizeof(float) * channel);
    // calculate mean
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            for (k = 0; k < hw; ++k) {
                mean[j] += input_0[i * chw + j * hw + k];
            }
        }
    }
    for (i = 0; i < channel; ++i) {
        mean[i] /= nhw;
    }
    // calcualte variance
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            for (k = 0; k < hw; ++k) {
                variance[j] += pow(input_0[i * chw + j * hw + k] - mean[j], 2);
            }
        }
    }
    for (i = 0; i < channel; ++i) {
        variance[i] /= nhw;
    }
    // scale shift mean variance
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            std = sqrt(variance[j] + batchnorm_op->eps);
            for (k = 0; k < hw; ++k) {
                // output_0[i * chw + j * hw + k] = (input_0[i * chw + j * hw + k] - mean[j]) / std
                // * input_1[j] + input_2[j];
                output_0[i * chw + j * hw + k] = (input_0[i * chw + j * hw + k] - mean[j]) / std;
            }
        }
    }
}

// already cal mean and  var
//  mean : input_1
//  var : input_2
static void op_default_batchnorm_average_run(op_t *op)
{
    const op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    int i, j, k, idx;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    int channel = op->input_tensors[0]->shape.dim[1];
    float std;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const float *input_2 = mem_cpu_data(op->input_tensors[2]->mem);
    // const float* input_3 = mem_cpu_data(op->input_tensors[3]->mem);
    // const float* input_4 = mem_cpu_data(op->input_tensors[4]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < channel; ++j) {
            std = sqrt(input_2[j] + batchnorm_op->eps);
            for (k = 0; k < hw; ++k) {
                output_0[k] = (input_0[k] - input_1[j]) / std;
            }
            input_0 += hw;
            output_0 += hw;
        }
    }
}

void op_default_batchnorm_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    // todo
    int channel = op->input_tensors[0]->shape.dim[1];
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (op->input_size == 1) {
            if (((op_batchnorm_t *)op)->mean == NULL) {
                ((op_batchnorm_t *)op)->mean = mem_new(cpu_mem_tp);
            }
            mem_alloc(((op_batchnorm_t *)op)->mean, channel * sizeof(float));
            if (((op_batchnorm_t *)op)->variance == NULL) {
                ((op_batchnorm_t *)op)->variance = mem_new(cpu_mem_tp);
            }
            mem_alloc(((op_batchnorm_t *)op)->variance, channel * sizeof(float));
            op->run_func = op_default_batchnorm_run;
        } else {
            op->run_func = op_default_batchnorm_average_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
