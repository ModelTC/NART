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

#include "art/cuda/cuda_mem.h"
#include "art/data_type.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    op_t o;
    uint32_t dtype;
} op_clip_cast_t;
op_clip_cast_t *op_cuda_clip_cast_tp_alloc(workspace_t *ws);
void op_cuda_clip_cast_tp_config(op_t *op);
void op_cuda_clip_cast_tp_prepare(op_t *op);
void op_cuda_clip_cast_tp_destroy(op_t *op);
void op_cuda_clip_cast_tp_dealloc(op_t *op);

#ifdef __cplusplus
} // extern "C"
#endif

op_clip_cast_t *op_cuda_clip_cast_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_clip_cast_t *res = (op_clip_cast_t *)malloc(sizeof(op_clip_cast_t));
    memset(res, 0, sizeof(op_clip_cast_t));
    return res;
}

void op_cuda_clip_cast_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_CAST_DTYPE, dtUINT32, &((op_clip_cast_t *)op)->dtype));
}

void op_cuda_clip_cast_tp_destroy(op_t *op) { (void)op; }

void op_cuda_clip_cast_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T, typename D>
__global__ void
op_cuda_clip_cast_kernel(D *c, T *a, const T *minimum, const T *maximum, size_t size)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        a[i] = max(min(a[i], *maximum), *minimum);
        c[i] = (D)a[i];
    }
}

template <typename T, typename D> static void op_cuda_clip_cast_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    T *input_0 = (T *)mem_data(op->input_tensors[0]->mem);
    D *output_0 = (D *)mem_data(op->output_tensors[0].mem);
    if (op->input_size == 3) {
        const T *min = (const T *)mem_data(op->input_tensors[1]->mem);
        const T *max = (const T *)mem_data(op->input_tensors[2]->mem);
        op_cuda_clip_cast_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, min, max, count);
    } else {
        CHECK(false);
    }
}

void op_cuda_clip_cast_tp_prepare(op_t *op)
{
    op_clip_cast_t *clip_cast_op = (op_clip_cast_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    // change output dtype
    uint32_t o_dtype = op->output_tensors[0].dtype = clip_cast_op->dtype;
    uint32_t i_dtype = op->input_tensors[0]->dtype;
    tensor_alloc(&op->output_tensors[0]);

    if (i_dtype == dtFLOAT32 && o_dtype == dtUINT8) {
        op->run_func = op_cuda_clip_cast_run<float, uint8_t>;
    } else if (i_dtype == dtINT32 && o_dtype == dtUINT8) {
        op->run_func = op_cuda_clip_cast_run<int32_t, uint8_t>;
    } else if (i_dtype == dtINT32 && o_dtype == dtINT8) {
        op->run_func = op_cuda_clip_cast_run<int32_t, int8_t>;
    } else {
        CHECK(false);
    }
}
