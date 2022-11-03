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

typedef enum { NONE = 0, MUL = 1, ADD = 2 } reduction_type;

typedef struct {
    op_t o;
    reduction_type reduction;
} op_scatternd_t;

op_scatternd_t *op_default_scatternd_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_scatternd_t *res = (op_scatternd_t *)malloc(sizeof(op_scatternd_t));
    memset(res, 0, sizeof(op_scatternd_t));
    return res;
}

void op_default_scatternd_tp_config(op_t *op)
{
    char *reduction;
    CHECK(op_setting_single_get(op, SETTING_SCATTERND_REDUCTION, dtSTR, &reduction));
    if (!strcmp(reduction, "")) {
        ((op_scatternd_t *)op)->reduction = NONE;
    } else if (!strcmp(reduction, "add")) {
        ((op_scatternd_t *)op)->reduction = ADD;
    } else if (!strcmp(reduction, "mul")) {
        ((op_scatternd_t *)op)->reduction = MUL;
    }
}

void op_default_scatternd_tp_destroy(op_t *op) { (void)op; }

void op_default_scatternd_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_scatternd_run(op_t *op)
{
    size_t i;
    int j;
    const float *data = mem_cpu_data(op->input_tensors[0]->mem);
    const float *indices = mem_cpu_data(op->input_tensors[1]->mem);
    const float *updates = mem_cpu_data(op->input_tensors[2]->mem);
    float *output = mem_cpu_data(op->output_tensors[0].mem);
    size_t count = shape_count(&op->input_tensors[0]->shape);
    size_t indices_count = shape_count(&op->input_tensors[1]->shape);
    size_t dim_size = op->input_tensors[0]->shape.dim_size;
    shape_t *shape = &op->input_tensors[0]->shape;
    size_t stride[MAX_DIM] = { 0 };
    stride[dim_size - 1] = 1;
    for (j = dim_size - 2; j > 0; j--) {
        stride[j] = stride[j + 1] * shape->dim[j + 1];
    }
    memcpy(output, data, sizeof(float) * count);
    for (i = 0; i < indices_count / dim_size; i++) {
        size_t offset = 0;
        for (j = dim_size - 1; j >= 0; j--) {
            offset += stride[j] * (int)indices[i * dim_size + j];
        }
        output[offset] = updates[i];
    }
}

void op_default_scatternd_tp_prepare(op_t *op)
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
        op->run_func = op_default_scatternd_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
