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

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    op_t o;
    int32_t axis;
} op_gather_t;
op_gather_t *op_default_gather_tp_alloc(workspace_t *ws);
void op_default_gather_tp_config(op_t *op);
void op_default_gather_tp_destroy(op_t *op);
void op_default_gather_tp_dealloc(op_t *op);
void op_default_gather_tp_prepare(op_t *op);

#ifdef __cplusplus
} // extern "C"
#endif

op_gather_t *op_default_gather_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_gather_t *res = (op_gather_t *)malloc(sizeof(op_gather_t));
    memset(res, 0, sizeof(op_gather_t));
    return res;
}

void op_default_gather_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_GATHER_AXIS, dtINT32, &((op_gather_t *)op)->axis));
}

void op_default_gather_tp_destroy(op_t *op) { (void)op; }

void op_default_gather_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T1, typename T2> static void op_default_gather_run(op_t *op)
{
    op_gather_t *gather_op = (op_gather_t *)op;
    size_t count = shape_count(&op->output_tensors[0].shape);
    const T1 *input_0 = (const T1 *)mem_cpu_data(op->input_tensors[0]->mem);
    const T2 *input_1 = (const T2 *)mem_cpu_data(op->input_tensors[1]->mem);
    T1 *output_0 = (T1 *)mem_cpu_data(op->output_tensors[0].mem);
    size_t i, j;
    int axis = gather_op->axis;
    axis = axis >= 0 ? axis : axis + op->input_tensors[0]->shape.dim_size;
    size_t axis_cnt = op->input_tensors[0]->shape.dim[axis];
    size_t inner_cnt = shape_part_count(&op->input_tensors[0]->shape, axis) / axis_cnt;
    size_t indices_cnt = shape_count(&op->input_tensors[1]->shape);
    size_t outer_cnt = shape_count(&op->input_tensors[0]->shape) / inner_cnt / axis_cnt;
    for (i = 0; i < outer_cnt; i++) {
        for (j = 0; j < indices_cnt; j++) {
            int ind = input_1[j];
            memcpy(
                output_0 + i * indices_cnt * inner_cnt + j * inner_cnt,
                input_0 + i * axis_cnt * inner_cnt + ind * inner_cnt, inner_cnt * sizeof(T1));
        }
    }
}

void op_default_gather_tp_prepare(op_t *op)
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
        switch (op->input_tensors[1]->dtype) {
        case dtINT64:
            op->run_func = op_default_gather_run<float, int64_t>;
            break;
        case dtFLOAT32:
            op->run_func = op_default_gather_run<float, float>;
            break;
        default:
            CHECK(false);
            break;
        }
        break;
    case dtINT64:
        switch (op->input_tensors[1]->dtype) {
        case dtINT64:
            op->run_func = op_default_gather_run<int64_t, int64_t>;
            break;
        case dtFLOAT32:
            op->run_func = op_default_gather_run<int64_t, float>;
            break;
        default:
            CHECK(false);
            break;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
