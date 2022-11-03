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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/tensor.h"

// gridsample mode
#define MODE_BILINEAR 1
#define MODE_NEREAST  2

// gridsample padding
#define PADDING_ZEROS      1
#define PADDING_BORDER     2
#define PADDING_REFLECTION 3

typedef struct {
    op_t o;
    int32_t mode;
    int32_t padding;
    int32_t align_corners;
} op_gridsample_t;

op_gridsample_t *op_default_gridsample_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_gridsample_t *res = (op_gridsample_t *)malloc(sizeof(op_gridsample_t));
    memset(res, 0, sizeof(op_gridsample_t));
    return res;
}

void op_default_gridsample_tp_config(op_t *op)
{
    op_gridsample_t *gridsample_op = (op_gridsample_t *)op;
    char *str;
    CHECK(op_setting_single_get(op, SETTING_GRIDSAMPLE_MODE, dtINT32, &(gridsample_op->mode)));
    CHECK(op_setting_single_get(
        op, SETTING_GRIDSAMPLE_PADDING_MODE, dtINT32, &(gridsample_op->padding)));
    CHECK(op_setting_single_get(
        op, SETTING_GRIDSAMPLE_ALIGN_CORNERS, dtBOOL, &(gridsample_op->align_corners)));
}

void op_default_gridsample_tp_destroy(op_t *op) { (void)op; }

void op_default_gridsample_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

#define SCALE_FACTOR(x, size) (((x) + 1) / 2 * (size)-0.5)

#define SCALE_FACTOR_ALIGN_CORNER(x, size) (((x) + 1) / 2 * ((size)-1))

#define TENSOR_GET_VALUE3D(ptr, c, h, w, ic, ih, iw) ((ptr)[(ic) * (h) * (w) + (ih) * (w) + (iw)])

#define TENSOR_GET_VALUE3D_PAD_ZEROS(ptr, c, h, w, ic, ih, iw)                       \
    (((ic) >= (c) || (ih) >= (h) || (iw) >= (w) || (ic) < 0 || (ih) < 0 || (iw) < 0) \
         ? 0                                                                         \
         : TENSOR_GET_VALUE3D(ptr, c, h, w, ic, ih, iw))

#define CLIP_NONE(x, low, high) (x)

#define CLIP(x, low, high) ((x) > (high) ? (high) : ((x) < (low) ? (low) : (x)))

#define TENSOR_GET_VALUE3D_PAD_BORDER(ptr, c, h, w, ic, ih, iw) \
    TENSOR_GET_VALUE3D(                                         \
        ptr, c, h, w, (ic >= c) ? c - 1 : ic, ih >= h ? h - 1 : ih, iw >= h ? h - 1 : ih)

#define REGISTER_GRIDSAMPLE_IMPL(func, SCALE_FACTOR_IMPL, CLIP_IMPL, GET_VALUE_IMPL)       \
    static void func(op_t *op)                                                             \
    {                                                                                      \
        const float *input = (const float *)mem_cpu_data(op->input_tensors[0]->mem);       \
        const float *grid = (const float *)mem_cpu_data(op->input_tensors[1]->mem);        \
        float *output = (float *)mem_cpu_data(op->output_tensors[0].mem);                  \
        const float *input_offset, *grid_offset;                                           \
        float *output_offset;                                                              \
                                                                                           \
        int i_h = op->input_tensors[0]->shape.dim[2];                                      \
        int i_w = op->input_tensors[0]->shape.dim[3];                                      \
                                                                                           \
        int o_n = op->output_tensors[0].shape.dim[0];                                      \
        int o_c = op->output_tensors[0].shape.dim[1];                                      \
        int o_h = op->output_tensors[0].shape.dim[2];                                      \
        int o_w = op->output_tensors[0].shape.dim[3];                                      \
        int n, c, h, w;                                                                    \
                                                                                           \
        for (n = 0; n < o_n; n++) {                                                        \
            for (c = 0; c < o_c; c++) {                                                    \
                grid_offset = grid + n * o_h * o_w * 2;                                    \
                input_offset = input + n * o_c * i_h * i_w + c * i_h * i_w;                \
                output_offset = output + n * o_c * o_h * o_w + c * o_h * o_w;              \
                for (h = 0; h < o_h; h++) {                                                \
                    for (w = 0; w < o_w; w++) {                                            \
                        float x = grid_offset[0];                                          \
                        float y = grid_offset[1];                                          \
                        grid_offset += 2;                                                  \
                                                                                           \
                        x = SCALE_FACTOR_IMPL(x, i_w);                                     \
                        y = SCALE_FACTOR_IMPL(y, i_h);                                     \
                                                                                           \
                        int x_nw = floor(x);                                               \
                        int y_nw = floor(y);                                               \
                        int x_ne = x_nw + 1;                                               \
                        int y_ne = y_nw;                                                   \
                        int x_sw = x_nw;                                                   \
                        int y_sw = y_nw + 1;                                               \
                        int x_se = x_nw + 1;                                               \
                        int y_se = y_nw + 1;                                               \
                                                                                           \
                        float s_se = (x - x_nw) * (y - y_nw);                              \
                        float s_sw = (x_ne - x) * (y - y_ne);                              \
                        float s_ne = (x - x_sw) * (y_sw - y);                              \
                        float s_nw = (x_se - x) * (y_se - y);                              \
                                                                                           \
                        float v_nw = GET_VALUE_IMPL(                                       \
                            input_offset, 1, i_h, i_w, 0, CLIP_IMPL(y_nw, 0, i_h - 1),     \
                            CLIP_IMPL(x_nw, 0, i_w - 1));                                  \
                        float v_ne = GET_VALUE_IMPL(                                       \
                            input_offset, 1, i_h, i_w, 0, CLIP_IMPL(y_ne, 0, i_h - 1),     \
                            CLIP_IMPL(x_ne, 0, i_w - 1));                                  \
                        float v_sw = GET_VALUE_IMPL(                                       \
                            input_offset, 1, i_h, i_w, 0, CLIP_IMPL(y_sw, 0, i_h - 1),     \
                            CLIP_IMPL(x_sw, 0, i_w - 1));                                  \
                        float v_se = GET_VALUE_IMPL(                                       \
                            input_offset, 1, i_h, i_w, 0, CLIP_IMPL(y_se, 0, i_h - 1),     \
                            CLIP_IMPL(x_se, 0, i_w - 1));                                  \
                                                                                           \
                        float val = s_nw * v_nw + s_ne * v_ne + s_sw * v_sw + s_se * v_se; \
                        *output_offset++ = val;                                            \
                    }                                                                      \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
    }

REGISTER_GRIDSAMPLE_IMPL(
    op_gridsample_4d_bilinear_pad_zero, SCALE_FACTOR, CLIP_NONE, TENSOR_GET_VALUE3D_PAD_ZEROS);
REGISTER_GRIDSAMPLE_IMPL(
    op_gridsample_4d_bilinear_pad_border, SCALE_FACTOR, CLIP, TENSOR_GET_VALUE3D);
REGISTER_GRIDSAMPLE_IMPL(
    op_gridsample_4d_bilinear_pad_zero_align_corner, SCALE_FACTOR_ALIGN_CORNER, CLIP_NONE,
    TENSOR_GET_VALUE3D_PAD_ZEROS);
REGISTER_GRIDSAMPLE_IMPL(
    op_gridsample_4d_bilinear_pad_border_align_corner, SCALE_FACTOR_ALIGN_CORNER, CLIP,
    TENSOR_GET_VALUE3D);

void op_default_gridsample_tp_prepare(op_t *op)
{
    op_gridsample_t *gridsample_op = (op_gridsample_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (gridsample_op->mode == GRIDSAMPLE_MODE_BILINEAR) {
            if (gridsample_op->align_corners) {
                if (gridsample_op->padding == GRIDSAMPLE_PADDING_ZEROS) {
                    op->run_func = op_gridsample_4d_bilinear_pad_zero_align_corner;
                } else if (gridsample_op->padding == GRIDSAMPLE_PADDING_BORDER) {
                    op->run_func = op_gridsample_4d_bilinear_pad_border_align_corner;
                } else {
                    LOG(error,
                        "unspoorted padding method for gridsample: `%d`, should be `zeros` or "
                        "`border`\n",
                        gridsample_op->padding);
                }
            } else {
                if (gridsample_op->padding == GRIDSAMPLE_PADDING_ZEROS) {
                    op->run_func = op_gridsample_4d_bilinear_pad_zero;
                } else if (gridsample_op->padding == GRIDSAMPLE_PADDING_BORDER) {
                    op->run_func = op_gridsample_4d_bilinear_pad_border;
                } else {
                    LOG(error,
                        "unspoorted padding method for gridsample: `%d`, should be `zeros` or "
                        "`border`\n",
                        gridsample_op->padding);
                }
            }
        } else {
            LOG(error, "unspoorted interpolate mode for gridsample: `%d`, should be `bilinear`\n",
                gridsample_op->mode);
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
