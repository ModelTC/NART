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

#include <string.h>

#include "art/log.h"
#include "art/tensor.h"

tensor_t *tensor_new(const mem_tp *tp, uint32_t dtype)
{
    tensor_t *res = (tensor_t *)malloc(sizeof(tensor_t));
    memset(res, 0, sizeof(tensor_t));
    res->mem = mem_new(tp);
    res->dtype = dtype;
    res->shape.dim_size = 4;
    res->shape.channel_axis = 1;
    res->with_transform = 0;
    return res;
}

void tensor_set_name(tensor_t *tensor, const char *name)
{
    if (tensor->name) {
        free(tensor->name);
        tensor->name = NULL;
    }
    if (NULL == name) {
        tensor->name = NULL;
        return;
    }
    char *n = (char *)realloc(tensor->name, strlen(name) + 1);
    CHECK_NE(NULL, n);
    strcpy(n, name);
    tensor->name = n;
}

void tensor_free(tensor_t *tensor)
{
    if (NULL != tensor->mem) {
        mem_delete(tensor->mem);
        tensor->mem = NULL;
    }
    if (NULL != tensor->name) {
        free(tensor->name);
        tensor->name = NULL;
    }
}

void tensor_delete(tensor_t *tensor)
{
    tensor_free(tensor);
    free(tensor);
}

bool tensor_shape_eq(const shape_t *a, const shape_t *b)
{
    bool res = true;
    if (a->dim_size == b->dim_size && a->batch_axis == b->batch_axis
        && a->channel_axis == b->channel_axis) {
        int i;
        for (i = 0; i < a->dim_size; ++i) {
            if (a->dim[i] != b->dim[i])
                return false;
        }
    } else {
        return false;
    }
    return res;
}

bool tensor_reshape(tensor_t *tensor, const int shape_size, const int shape[])
{
    int infer_idx = -1;
    uint64_t sz = 1;
    uint64_t szz = 1;
    uint32_t temp_sp[MAX_DIM];
    int i;
    for (i = 0; i < shape_size; ++i) {
        if (i < tensor->shape.dim_size)
            sz *= tensor->shape.dim[i];
        switch (shape[i]) {
        case 0:
            CHECK_LT(i, tensor->shape.dim_size);
            temp_sp[i] = tensor->shape.dim[i];
            szz *= temp_sp[i];
            break;
        case -1:
            if (-1 != infer_idx) {
                fprintf(stderr, "Tensor reshape failed, cannot specify -1 more than once");
                return false;
            }
            infer_idx = i;
            break;
        default:
            temp_sp[i] = shape[i];
            szz *= temp_sp[i];
            break;
        }
    }
    for (; i < tensor->shape.dim_size; ++i) {
        sz *= tensor->shape.dim[i];
    }
    if (infer_idx >= 0) {
        CHECK_EQ(0, sz % szz);
        temp_sp[infer_idx] = sz / szz;
    }

    if (tensor->shape.batch_axis > shape_size) {
        tensor->shape.batch_axis = -1;
        LOG(warn, "batch_axis greater than dim_size, set -1\n");
    }

    if (tensor->shape.channel_axis > shape_size) {
        tensor->shape.channel_axis = -1;
        LOG(warn, "channel_axis greater than dim_size, set -1\n");
    }

    sz = shape_count(&tensor->shape);
    tensor->shape.dim_size = shape_size;
    memcpy(tensor->shape.dim, temp_sp, sizeof(uint32_t) * tensor->shape.dim_size);
    szz = shape_count(&tensor->shape);
    if (sz != 0 && szz > sz) {
        tensor_alloc(tensor);
    }
    return true;
}

void tensor_share(tensor_t *share_from, tensor_t *share_to)
{
    if (NULL == share_from || NULL == share_to)
        return;
    mem_delete(share_to->mem);
    share_to->mem = share_from->mem;
    share_to->mem->refcount++;
}
