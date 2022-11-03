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

typedef struct {
    op_t o;
    uint32_t axis;
    size_t slice_len;
    uint32_t *slice_points;
} op_slice_t;

op_slice_t *op_default_slice_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_slice_t *res = (op_slice_t *)malloc(sizeof(op_slice_t));
    memset(res, 0, sizeof(op_slice_t));
    return res;
}

void op_default_slice_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_SLICE_AXIS, dtUINT32, &((op_slice_t *)op)->axis));
    CHECK(op_setting_array_get(
        op, SETTING_SLICE_POINT, dtUINT32, &((op_slice_t *)op)->slice_len,
        &((op_slice_t *)op)->slice_points));
}

void op_default_slice_tp_destroy(op_t *op) { (void)op; }

void op_default_slice_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_slice_run(op_t *op)
{
    int i, j;
    int batch_size = op->input_tensors[0]->shape.dim[0];
    size_t count;
    uint32_t axis = ((op_slice_t *)op)->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_size = shape_part_count(&op->input_tensors[0]->shape, axis) / ch;
    size_t outer_size = shape_count(&op->input_tensors[0]->shape) / ch / inner_size;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output = NULL;
    for (i = 0; i < outer_size; ++i) {
        for (j = 0; j < op->output_size; ++j) {
            output = mem_cpu_data(op->output_tensors[j].mem);
            count = inner_size * op->output_tensors[j].shape.dim[axis];
            memcpy(output + count * i, input_0, sizeof(float) * count);
            input_0 += count;
        }
    }
}

void op_default_slice_tp_prepare(op_t *op)
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
        op->run_func = op_default_slice_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
