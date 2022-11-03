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
    uint8_t method; // 0: down 1: up
    uint32_t sample;
} op_subpixel_t;

op_subpixel_t *op_default_subpixel_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_subpixel_t *res = (op_subpixel_t *)malloc(sizeof(op_subpixel_t));
    memset(res, 0, sizeof(op_subpixel_t));
    return res;
}

void op_default_subpixel_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_SUBPIXEL_METHOD, dtUINT8, &((op_subpixel_t *)op)->method));
    CHECK(op_setting_single_get(
        op, SETTING_SUBPIXEL_SAMPLE, dtUINT32, &((op_subpixel_t *)op)->sample));
}

void op_default_subpixel_tp_destroy(op_t *op) { (void)op; }

void op_default_subpixel_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_subpixel_down_run_f32(op_t *op)
{
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t channel = op->input_tensors[0]->shape.dim[1];
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];
    size_t sample = ((op_subpixel_t *)op)->sample;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    channel *= sample * sample;
    height /= sample;
    width /= sample;

    size_t n, c, h, w;
    size_t bottom_channel = channel / (sample * sample);
    size_t bottom_height = height * sample;
    size_t bottom_width = width * sample;

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            size_t bottom_c = c / (sample * sample);
            size_t sub_idx = c % (sample * sample);
            size_t sub_h = sub_idx / sample;
            size_t sub_w = sub_idx % sample;
            for (h = 0; h < height; ++h) {
                size_t bottom_h = h * sample + sub_h;
                for (w = 0; w < width; ++w) {
                    size_t bottom_w = w * sample + sub_w;
                    size_t idx = ((n * channel + c) * height + h) * width + w;
                    // size_t old_idx = ((n * bottom_channel + bottom_c) * bottom_height + bottom_h)
                    // * bottom_width + bottom_w;
                    size_t old_idx = ((n * bottom_channel + bottom_c) * bottom_height + bottom_h)
                            * bottom_width
                        + bottom_w;
                    output_0[idx] = input_0[old_idx];
                }
            }
        }
    }
}

static void op_default_subpixel_up_run_f32(op_t *op)
{
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t channel = op->input_tensors[0]->shape.dim[1];
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];
    size_t sample = ((op_subpixel_t *)op)->sample;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    channel /= sample * sample;
    height *= sample;
    width *= sample;

    size_t n, c, h, w;
    size_t bottom_channel = channel * (sample * sample);
    size_t bottom_height = height / sample;
    size_t bottom_width = width / sample;

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            for (h = 0; h < height; ++h) {
                size_t bottom_h = h / sample;
                for (w = 0; w < width; ++w) {
                    size_t bottom_w = w / sample;
                    size_t sub_h = h % sample;
                    size_t sub_w = w % sample;
                    // size_t bottom_c = (sub_h * sample + sub_w) * channel + c;
                    size_t bottom_c = (c * sample + sub_h) * sample + sub_w;
                    size_t idx = ((n * channel + c) * height + h) * width + w;
                    size_t old_idx = ((n * bottom_channel + bottom_c) * bottom_height + bottom_h)
                            * bottom_width
                        + bottom_w;
                    output_0[idx] = input_0[old_idx];
                }
            }
        }
    }
}

static void op_default_subpixel_down_run_u8(op_t *op)
{
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t channel = op->input_tensors[0]->shape.dim[1];
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];
    size_t sample = ((op_subpixel_t *)op)->sample;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    channel *= sample * sample;
    height /= sample;
    width /= sample;

    size_t n, c, h, w;
    size_t bottom_channel = channel / (sample * sample);
    size_t bottom_height = height * sample;
    size_t bottom_width = width * sample;

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            size_t bottom_c = c % (channel / (sample * sample));
            size_t sub_idx = c / bottom_channel;
            size_t sub_h = sub_idx / sample;
            size_t sub_w = sub_idx % sample;
            for (h = 0; h < height; ++h) {
                size_t bottom_h = h * sample + sub_h;
                for (w = 0; w < width; ++w) {
                    size_t bottom_w = w * sample + sub_w;
                    size_t idx = ((n * channel + c) * height + h) * width + w;
                    size_t old_idx = ((n * bottom_channel + bottom_c) * bottom_height + bottom_h)
                            * bottom_width
                        + bottom_w;
                    output_0[idx] = input_0[old_idx];
                }
            }
        }
    }
}

static void op_default_subpixel_up_run_u8(op_t *op)
{
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t channel = op->input_tensors[0]->shape.dim[1];
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];
    size_t sample = ((op_subpixel_t *)op)->sample;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    channel /= sample * sample;
    height *= sample;
    width *= sample;

    size_t n, c, h, w;
    size_t bottom_channel = channel * (sample * sample);
    size_t bottom_height = height / sample;
    size_t bottom_width = width / sample;

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            for (h = 0; h < height; ++h) {
                size_t bottom_h = h / sample;
                for (w = 0; w < width; ++w) {
                    size_t bottom_w = w / sample;
                    size_t sub_h = h % sample;
                    size_t sub_w = w % sample;
                    // size_t bottom_c = (sub_h * sample + sub_w) * channel + c;
                    size_t bottom_c = (c * sample + sub_h) * sample + sub_w;
                    size_t idx = ((n * channel + c) * height + h) * width + w;
                    size_t old_idx = ((n * bottom_channel + bottom_c) * bottom_height + bottom_h)
                            * bottom_width
                        + bottom_w;
                    output_0[idx] = input_0[old_idx];
                }
            }
        }
    }
}

void op_default_subpixel_tp_prepare(op_t *op)
{
    int i;
    op_subpixel_t *subpixel_op = (op_subpixel_t *)op;
    size_t method = subpixel_op->method;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (0 == method) {
            op->run_func = op_default_subpixel_down_run_f32;
        } else {
            op->run_func = op_default_subpixel_up_run_f32;
        }
        break;
    case dtUINT8:
        if (0 == method) {
            op->run_func = op_default_subpixel_down_run_u8;
        } else {
            op->run_func = op_default_subpixel_up_run_u8;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
