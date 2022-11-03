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
#include "art/tensor.h"

typedef struct {
    op_t o;
    int32_t *dims;
    size_t dim_size;
    int32_t axis;
    int32_t num_axes;
} op_reshape_t;

op_reshape_t *op_default_reshape_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_reshape_t *res = (op_reshape_t *)malloc(sizeof(op_reshape_t));
    memset(res, 0, sizeof(op_reshape_t));
    return res;
}

void op_default_reshape_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_RESHAPE_AXIS, dtINT32, &((op_reshape_t *)op)->axis));
    CHECK(op_setting_single_get(
        op, SETTING_RESHAPE_NUM_AXES, dtINT32, &((op_reshape_t *)op)->num_axes));
    CHECK(op_setting_array_get(
        op, SETTING_RESHAPE_DIMS, dtINT32, &((op_reshape_t *)op)->dim_size,
        &((op_reshape_t *)op)->dims));
}

void op_default_reshape_tp_destroy(op_t *op) { (void)op; }

void op_default_reshape_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_reshape_run(op_t *op) { (void)op; }

void op_default_reshape_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtUNKNOWN:
        CHECK(false);
        break;
    default:
        op->run_func = op_default_reshape_run;
        break;
    }
}
