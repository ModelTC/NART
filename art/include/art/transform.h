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

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RAWDATA_TO_TENSOR     0
#define RESIZE_TO_TENSOR      1
#define AFFINE_TO_TENSOR      2
#define PERSPECTIVE_TO_TENSOR 3
#define TO_TENSOR_MASK        0xF
#define CHANNEL_NORM          (1 << 4)
#define EQUALHIST             (1 << 5)

typedef struct frame_t {
    int32_t fmt;
    int32_t width;
    int32_t height;
    int64_t pts;

    int32_t plane_num;
    uint8_t *plane[4];
    int32_t stride[4];
    void *data;
    size_t size;
    void *extra_info;
} frame_t;

typedef struct pixel_t {
    float r;
    float g;
    float b;
} pixel_t;

typedef struct transform_param_t {
    const char *tensor_name;
    uint32_t operators;
    size_t batch;
    frame_t **frames;
    int32_t tensor_type;
    int32_t (*rois)[4]; // x, y, w, h
    // the destination roi, aka, the paste region.
    int32_t (*dst_rois)[4]; // x, y, w, h
    float (*mats)[3][3];
    pixel_t means;
    pixel_t stds;
    pixel_t paddings;
} transform_param_t;

typedef struct transform_tensor_t {
    tensor_t tensor;
    size_t capacity;
    transform_param_t param;
} transform_tensor_t;

#ifdef __cplusplus
}
#endif
#endif // TRANSFORM_H
