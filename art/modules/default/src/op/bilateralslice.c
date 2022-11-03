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

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef struct {
    op_t o;
    uint32_t coe;
    bool offset;
    mem_t *temp_data;

} op_bilateralslice_t;

op_bilateralslice_t *op_default_bilateralslice_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_bilateralslice_t *res = (op_bilateralslice_t *)malloc(sizeof(op_bilateralslice_t));
    memset(res, 0, sizeof(op_bilateralslice_t));
    return res;
}

void op_default_bilateralslice_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_BILATERALSLICE_COE, dtUINT32, &((op_bilateralslice_t *)op)->coe));
    CHECK(op_setting_single_get(
        op, SETTING_BILATERALSLICE_OFFSET, dtBOOL, &((op_bilateralslice_t *)op)->offset));
}

void op_default_bilateralslice_tp_destroy(op_t *op)
{
    if (((op_bilateralslice_t *)op)->temp_data)
        mem_delete(((op_bilateralslice_t *)op)->temp_data);
    ((op_bilateralslice_t *)op)->temp_data = NULL;
}

void op_default_bilateralslice_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_bilateralslice_simple_run(op_t *op)
{ // B C H W
    op_bilateralslice_t *tri_op = (op_bilateralslice_t *)op;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    int i_c = op->input_tensors[0]->shape.dim[1];
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];

    int g_c = op->input_tensors[1]->shape.dim[1];
    int g_h = op->input_tensors[1]->shape.dim[2];
    int g_w = op->input_tensors[1]->shape.dim[3];

    const float scale_x = (float)g_w / i_w;
    const float scale_y = (float)g_h / i_h;

    const float *input = mem_cpu_data(op->input_tensors[0]->mem);
    const float *grid = mem_cpu_data(op->input_tensors[1]->mem);
    const float *guide = mem_cpu_data(op->input_tensors[2]->mem);
    float *output = mem_cpu_data(op->output_tensors[0].mem);

    float *tmp = mem_cpu_data(tri_op->temp_data);

    int coe = tri_op->coe;
    int dep = g_c / coe;

    bool has_offset = tri_op->offset;
    float val;
    int bs, i, j, k, l;
    for (i = 0; i < coe; i++) {
        int depth = dep * i;
        for (j = 0; j < i_h; j++) {
            float gy = (j + 0.5f) * scale_y;
            int fy = floor(gy - 0.5f);
            for (k = 0; k < i_w; k++) {
                val = 0;
                float gx = (k + 0.5f) * scale_x;
                float gz = guide[i_w * j + k] * dep;
                int fx = floor(gx - 0.5f);
                int fz = floor(gz - 0.5f);

                int x_left = max(min(fx, gx - 1), 0.0f);
                int x_right = max(min(fx + 1, gx - 1), 0.0f);
                float wx_0 = max(1.0 - fabs(fx + 0.5 - gx), 0.0f);
                float wx_1 = 1 - wx_0; // max(1.0 - fabs(fx + 1 + 0.5 - gx), 0.0f);

                int y_left = max(min(fy, g_h - 1), 0.0f);
                int y_right = max(min(fy + 1, g_h - 1), 0.0f);
                float wy_0 = max(1.0 - fabs(fy + 0.5 - gy), 0.0f);
                float wy_1 = 1 - wy_0; // max(1.0 - fabs(fy + 1 + 0.5 - gy), 0.0f);

                int z_left = max(min(fz, dep - 1), 0.0f);
                int z_right = max(min(fz + 1, dep - 1), 0.0f);

                float wz_0 = z_left + 0.5 - gz;
                wz_0 = wz_0 * wz_0;
                wz_0 = sqrt(wz_0 + 1e-8);
                wz_0 = max(1 - wz_0, 0.0f);

                float wz_1 = 1 - wz_0; // z_right + 0.5 - gz;
                //                wz_1 = wz_1 * wz_1;
                //                wz_1 = sqrt(wz_1 + 1e-8);
                //                wz_1 = max(1 - wz_1, 0.0f);

                int grid_idx_000 = ((z_left + depth) * g_h + y_left) * g_w + x_left;
                int grid_idx_001 = ((z_left + depth) * g_h + y_left) * g_w + x_right;

                int grid_idx_010 = ((z_left + depth) * g_h + y_right) * g_w + x_left;
                int grid_idx_011 = ((z_left + depth) * g_h + y_right) * g_w + x_right;

                int grid_idx_100 = ((z_right + depth) * g_h + y_left) * g_w + x_left;
                int grid_idx_101 = ((z_right + depth) * g_h + y_left) * g_w + x_right;

                int grid_idx_110 = ((z_right + depth) * g_h + y_right) * g_w + x_left;
                int grid_idx_111 = ((z_right + depth) * g_h + y_right) * g_w + x_right;

                val += grid[grid_idx_000] * wx_0 * wy_0 * wz_0;
                val += grid[grid_idx_100] * wx_0 * wy_0 * wz_1;
                val += grid[grid_idx_010] * wx_0 * wy_1 * wz_0;
                val += grid[grid_idx_110] * wx_0 * wy_1 * wz_1;
                val += grid[grid_idx_001] * wx_1 * wy_0 * wz_0;
                val += grid[grid_idx_101] * wx_1 * wy_0 * wz_1;
                val += grid[grid_idx_011] * wx_1 * wy_1 * wz_0;
                val += grid[grid_idx_111] * wx_1 * wy_1 * wz_1;

                tmp[(i * i_h + j) * i_w + k] = val;
            }
        }
    }
    for (bs = 0; bs < (int)batch_size; bs++) {
        for (i = 0; i < i_h; i++) {
            for (j = 0; j < i_w; j++) {
                for (k = 0; k < i_c; k++) {
                    float sum = 0;
                    if (has_offset == true)
                        sum = tmp[((k + i_c * i_c) * i_h + i) * i_w + j];
                    for (l = 0; l < i_c; l++) {
                        sum += tmp[((l * i_c + k) * i_h + i) * i_w + j]
                            * input[(l * i_h + i) * i_w + j];
                    }
                    output[(k * i_h + i) * i_w + j] = sum;
                }
            }
        }
    }
}
void op_default_bilateralslice_tp_prepare(op_t *op)
{
    int i;
    op_bilateralslice_t *tri_op = (op_bilateralslice_t *)op;

    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    if (NULL == tri_op->temp_data)
        tri_op->temp_data = mem_new(cpu_mem_tp);

    mem_alloc(
        tri_op->temp_data,
        sizeof(float) * tri_op->coe * op->input_tensors[0]->shape.dim[2]
            * op->input_tensors[0]->shape.dim[3]);

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_bilateralslice_simple_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
