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
    int32_t max_disp;
    int32_t kernel_size;
    int32_t pad;
    int32_t single_direction;
} op_correlation1d_t;
op_correlation1d_t *op_default_correlation1d_tp_alloc(workspace_t *ws);
void op_default_correlation1d_tp_config(op_t *op);
void op_default_correlation1d_tp_destroy(op_t *op);
void op_default_correlation1d_tp_dealloc(op_t *op);
void op_default_correlation1d_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_correlation1d_t *op_default_correlation1d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_correlation1d_t *res = (op_correlation1d_t *)malloc(sizeof(op_correlation1d_t));
    memset(res, 0, sizeof(op_correlation1d_t));
    return res;
}

void op_default_correlation1d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION1D_MAX_DISPLACEMENT, dtINT32,
        &((op_correlation1d_t *)op)->max_disp));
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION1D_KERNEL_SIZE, dtINT32, &((op_correlation1d_t *)op)->kernel_size));
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION1D_PAD, dtINT32, &((op_correlation1d_t *)op)->pad));
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION1D_SINGLE_DIRECTION, dtINT32,
        &((op_correlation1d_t *)op)->single_direction));
}

void op_default_correlation1d_tp_destroy(op_t *op) { }

void op_default_correlation1d_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_correlation1d_run(op_t *op)
{
    op_correlation1d_t *corr_op = (op_correlation1d_t *)op;
    float *in0_data = (float *)mem_cpu_data(op->input_tensors[0]->mem);
    float *in1_data = (float *)mem_cpu_data(op->input_tensors[1]->mem);
    float *out_data = (float *)mem_cpu_data(op->output_tensors[0].mem);

    size_t in = op->input_tensors[0]->shape.dim[0];
    size_t ic = op->input_tensors[0]->shape.dim[1];
    size_t ih = op->input_tensors[0]->shape.dim[2];
    size_t iw = op->input_tensors[0]->shape.dim[3];

    size_t o_chw = op->output_tensors[0].shape.dim[1] * ih * iw;

    memset(out_data, 0, shape_count(&op->output_tensors[0].shape) * sizeof(float));
    size_t on, oc, oh, ow, r;
    for (on = 0; on < in; on++) {
        for (oc = 0; oc < op->output_tensors[0].shape.dim[1]; oc++) {
            for (oh = 0; oh < ih; oh++) {
                for (ow = oc + 1; ow < iw; ow++) {
                    float res = 0.;
                    for (r = 0; r < ic; r++) {
                        res += in0_data[on * ic * ih * iw + r * ih * iw + oh * iw + ow]
                            * in1_data[on * ic * ih * iw + r * ih * iw + oh * iw + (ow - oc - 1)];
                    }
                    out_data[on * o_chw + (corr_op->max_disp - oc) * ih * iw + oh * iw + ow]
                        = res / ic;
                }
            }
        }
    }
}

void op_default_correlation1d_tp_prepare(op_t *op)
{
    int i;
    op_correlation1d_t *conv_op = (op_correlation1d_t *)op;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_correlation1d_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
