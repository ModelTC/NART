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

#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data_type.h"
#include "log.h"
#include "mem.h"

#define MAX_DIM 8
#ifdef __cplusplus
extern "C" {
#endif

typedef struct shape_t {
    uint32_t dim[MAX_DIM];
    int32_t dim_size; /* default = 4 */
    int32_t batch_axis; /* default = 0 */
    int32_t channel_axis; /* default = 1 */
} shape_t;

typedef struct tensor_t {
    char *name;
    mem_t *mem;
    uint32_t dtype;
    shape_t shape;
    int32_t with_transform;
} tensor_t;

tensor_t *tensor_new(const mem_tp *tp, uint32_t dtype);

void tensor_free(tensor_t *tensor);

static inline int8_t shape_set_batch_axis(shape_t *shape, int8_t axis)
{
    if (axis >= shape->dim_size) {
        return shape->batch_axis;
    } else {
        shape->batch_axis = axis;
        return axis;
    }
}

static inline int8_t shape_set_channel_axis(shape_t *shape, int8_t axis)
{
    if (axis >= shape->dim_size) {
        return shape->channel_axis;
    } else {
        shape->channel_axis = axis;
        return axis;
    }
}

static inline size_t shape_count(const shape_t *shape)
{
    int i;
    size_t c = 1;
    if (0 == shape->dim_size)
        return 0;
    for (i = 0; i < shape->dim_size; ++i) {
        c *= shape->dim[i];
    }
    return c;
}

static inline size_t shape_count_per_batch(const shape_t *shape)
{
    size_t count = shape_count(shape);
    if (shape->batch_axis >= 0)
        return count / shape->dim[shape->batch_axis];
    return count;
}

// hp suggest that call it shape_part_count
static inline size_t shape_part_count(const shape_t *shape, int axis)
{
    int i;
    size_t c = 1;
    if (0 == shape->dim_size)
        return 0;
    int start_axis = axis >= 0 ? axis : shape->dim_size + axis;
    CHECK_GE(shape->dim_size, start_axis);
    for (i = start_axis; i < shape->dim_size; ++i) {
        c *= shape->dim[i];
    }
    return c;
}

static inline void tensor_alloc(tensor_t *tensor)
{
    if (NULL != tensor && tensor->dtype != dtUNKNOWN)
        mem_alloc(tensor->mem, datatype_sizeof(tensor->dtype) * shape_count(&tensor->shape));
}

static inline const char *tensor_name(tensor_t *tensor) { return tensor->name; }

void tensor_set_name(tensor_t *tesnor, const char *name);

void tensor_delete(tensor_t *tensor);

bool tensor_shape_eq(const shape_t *a, const shape_t *b);

bool tensor_reshape(tensor_t *tensor, const int shape_size, const int shape[]);

void tensor_dup(tensor_t *target, const tensor_t *from);

void tensor_share(tensor_t *share_from, tensor_t *share_to);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
