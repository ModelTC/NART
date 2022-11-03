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

typedef struct {
    op_t o;
    uint32_t mode;
    float value;
    int32_t *pads;
} op_pad_t;

op_pad_t *op_default_pad_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pad_t *res = (op_pad_t *)malloc(sizeof(op_pad_t));
    memset(res, 0, sizeof(op_pad_t));
    return res;
}

void op_default_pad_tp_config(op_t *op)
{
    size_t len;
    CHECK(op_setting_single_get(op, SETTING_PAD_MODE, dtUINT32, &((op_pad_t *)op)->mode));
    CHECK(op_setting_single_get(op, SETTING_PAD_VALUE, dtFLOAT32, &((op_pad_t *)op)->value));
    CHECK(op_setting_array_get(op, SETTING_PAD_PADS, dtINT32, &len, &((op_pad_t *)op)->pads));
    CHECK(8 == len);
    /* TODO:
    Only supprt NCHW now;
    High dimensions need to be added later?
    */
}

void op_default_pad_tp_destroy(op_t *op) { (void)op; }

void op_default_pad_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void PadImageConst(
    int batch_size, int i_c, int i_h, int i_w, int o_c, int o_h, int o_w, int32_t *pads, float val,
    const float *input_0, float *output_0)
{
    int n, c, h, w;
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < o_c; ++c) {
            int pc = c - pads[1];
            for (h = 0; h < o_h; ++h) {
                for (w = 0; w < o_w; ++w) {
                    int ph = h - pads[2];
                    int pw = w - pads[3];
                    output_0[h * o_w + w]
                        = (pc < 0 || ph < 0 || pw < 0 || pc >= i_c || ph >= i_h || pw >= i_w)
                        ? val
                        : input_0[ph * i_w + pw];
                }
            }
            if (pc >= 0 && pc < i_c)
                input_0 += i_h * i_w;
            output_0 += o_h * o_w;
        }
    }
}

void PadImageReflect(
    int batch_size, int i_c, int i_h, int i_w, int o_c, int o_h, int o_w, int32_t *pads,
    const float *input_0, float *output_0)
{
    int n, c, h, w;
    if (pads[2] >= 0 && pads[3] >= 0 && pads[6] >= 0 && pads[7] >= 0) {
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < o_c; ++c) {
                // Handle the valid region:
                for (h = -pads[2]; h < o_h - pads[6]; ++h) {
                    for (w = -pads[3]; w < o_w - pads[7]; ++w) {
                        output_0[h * o_w + w] = input_0[h * i_w + w];
                    }
                }
// Fixup areas where we need to reflect
#define X(h, w)                      \
    int ph = h - pads[2];            \
    int pw = w - pads[3];            \
    ph = fmax(ph, -ph);              \
    ph = fmin(ph, 2 * i_h - ph - 2); \
    pw = fmax(pw, -pw);              \
    pw = fmin(pw, 2 * i_w - pw - 2); \
    output_0[h * o_w + w] = input_0[ph * i_w + pw]

                // Top part
                for (h = 0; h < pads[2]; ++h) {
                    for (w = 0; w < o_w; ++w) {
                        X(h, w);
                    }
                }
                // Bottom part
                for (h = o_h - pads[6]; h < o_h; ++h) {
                    for (w = 0; w < o_w; ++w) {
                        X(h, w);
                    }
                }
                // Interior
                for (h = pads[2]; h < o_h - pads[6]; ++h) {
                    // Left
                    for (w = 0; w < pads[3]; ++w) {
                        X(h, w);
                    }
                    // Right
                    for (w = o_w - pads[7]; w < o_w; ++w) {
                        X(h, w);
                    }
                }
#undef X
                // Do offset.
                input_0 += i_h * i_w;
                output_0 += o_h * o_w;
            }
        }
    } else {
        for (n = 0; n < batch_size; ++n) {
            for (c = 0; c < o_c; ++c) {
                for (h = 0; h < o_h; ++h) {
                    for (w = 0; w < o_w; ++w) {
                        int ph = h - pads[2];
                        int pw = w - pads[3];
                        // max(h, -h) does reflection over 0
                        ph = fmax(ph, -ph);
                        // min(h, 2 * height - h - 2) does reflection over height.
                        ph = fmin(ph, 2 * i_h - ph - 2);
                        pw = fmax(pw, -pw);
                        pw = fmin(pw, 2 * i_w - pw - 2);
                        output_0[h * o_w + w] = input_0[ph * i_w + pw];
                    }
                }
                // Do offset.
                input_0 += i_h * i_w;
                output_0 += o_h * o_w;
            }
        }
    }
}

void PadImageEdge(
    int batch_size, int i_c, int i_h, int i_w, int o_c, int o_h, int o_w, int32_t *pads,
    const float *input_0, float *output_0)
{
    int n, c, h, w;
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < i_c; ++c) {
            for (h = 0; h < o_h; ++h) {
                for (w = 0; w < o_w; ++w) {
                    // Bounds to the right range.
                    int ph = fmin(i_h - 1, fmax(h - pads[2], 0));
                    int pw = fmin(i_w - 1, fmax(w - pads[3], 0));
                    output_0[h * o_w + w] = input_0[ph * i_w + pw];
                }
            }
            // Do offset.
            input_0 += i_h * i_w;
            output_0 += o_h * o_w;
        }
    }
}

static void op_default_pad_run(op_t *op)
{
    uint32_t mode = ((op_pad_t *)op)->mode;
    float val = ((op_pad_t *)op)->value;
    int32_t *pads;
    size_t len;
    CHECK(op_setting_array_get(op, SETTING_PAD_PADS, dtINT32, &len, &pads));
    int o_c, o_h, o_w, batch_size, i_h, i_w, i_c;
    o_c = op->output_tensors[0].shape.dim[1];
    o_h = op->output_tensors[0].shape.dim[2];
    o_w = op->output_tensors[0].shape.dim[3];
    i_c = op->input_tensors[0]->shape.dim[1];
    i_h = op->input_tensors[0]->shape.dim[2];
    i_w = op->input_tensors[0]->shape.dim[3];
    batch_size = op->input_tensors[0]->shape.dim[0];
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
#define CONSTANT 1
#define REFLECT  2
#define EDGE     3
    switch (mode) {
    case CONSTANT:
        PadImageConst(batch_size, i_c, i_h, i_w, o_c, o_h, o_w, pads, val, input_0, output_0);
        break;
    case REFLECT:
        PadImageReflect(batch_size, i_c, i_h, i_w, o_c, o_h, o_w, pads, input_0, output_0);
        break;
    case EDGE:
        PadImageEdge(batch_size, i_c, i_h, i_w, o_c, o_h, o_w, pads, input_0, output_0);
        break;
    default:
        break;
    }
#undef CONSTANT
#undef REFLECT
#undef EDGE
    (void)op;
}

void op_default_pad_tp_prepare(op_t *op)
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
        op->run_func = op_default_pad_run;
        break;
    }
}
