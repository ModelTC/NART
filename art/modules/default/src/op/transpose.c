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
} op_transpose_t;

op_transpose_t *op_default_transpose_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_transpose_t *res = (op_transpose_t *)malloc(sizeof(op_transpose_t));
    memset(res, 0, sizeof(op_transpose_t));
    return res;
}

void op_default_transpose_tp_config(op_t *op)
{
    size_t ndim;
    CHECK(op_setting_array_get(
        op, SETTING_TRANSPOSE_DIMS, dtINT32, &ndim, &((op_transpose_t *)op)->dims));
}

void op_default_transpose_tp_destroy(op_t *op) { (void)op; }

void op_default_transpose_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static inline int _check_dim(const uint32_t *shape_src, const int *axis, int dim_size)
{
    int order = -1;
    int i;
    for (i = 0; i < dim_size; ++i) {
        if (shape_src[axis[i]] == 1) {
            continue;
        }
        if (axis[i] < order) {
            return 0;
        }
        order = axis[i];
    }
    return 1;
}

static inline int mul(uint32_t *beg, uint32_t *end)
{
    int res = 1;
    while (beg != end) {
        res *= *beg;
        beg++;
    }
    return res;
}

static void op_default_transpose_run(op_t *op)
{
    op_transpose_t *transpose = (op_transpose_t *)op;
    int current[MAX_DIM];
    uint32_t transposed_dims[MAX_DIM];
    // int rev_axis[MAX_DIM];
    int32_t *axis = transpose->dims;
    uint32_t *dims = op->input_tensors[0]->shape.dim;

    int dim_size = op->input_tensors[0]->shape.dim_size;
    float *input = mem_cpu_data(op->input_tensors[0]->mem);
    float *dst_data = mem_cpu_data(op->output_tensors[0].mem);
    int i;
    for (i = 0; i < dim_size; ++i) {
        transposed_dims[i] = dims[axis[i]];
        // rev_axis[axis[i]] = i;
        current[i] = 0;
    }

    /* copy directly when no non-1 axis affected */
    /* skip the stupid opt */
    if (_check_dim(dims, axis, dim_size)) {
        memcpy(dst_data, input, sizeof(float) * shape_count(&op->input_tensors[0]->shape));
        return;
    }

    const int stride = mul(dims + axis[dim_size - 1] + 1, dims + dim_size);
    do {
        size_t bz = 0;
        for (i = 0; i < dim_size; ++i) {
            bz += current[i] * mul(dims + axis[i] + 1, dims + dim_size);
        }
        const float *src_data = input + bz;
        for (i = 0; i < (int)transposed_dims[dim_size - 1]; ++i) {
            *(dst_data++) = *src_data;
            src_data += stride;
        }
        current[dim_size - 2] += 1;
        for (i = (int)dim_size - 2; i >= 0; --i) {
            int d = current[i];
            if (d >= (int)transposed_dims[i]) {
                if (i == 0)
                    return;
                current[i] = 0;
                current[i - 1] += 1;
            }
        }
    } while (true);
}

void op_default_transpose_tp_prepare(op_t *op)
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
        op->run_func = op_default_transpose_run;
        break;
    }
}
