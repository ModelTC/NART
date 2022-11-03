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
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef struct {
    op_t o;
    uint32_t coe;
    bool offset;
    mem_t *temp_data;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_bilateralslice_t;

op_bilateralslice_t *op_quant_bilateralslice_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_bilateralslice_t *res = (op_bilateralslice_t *)malloc(sizeof(op_bilateralslice_t));
    memset(res, 0, sizeof(op_bilateralslice_t));
    return res;
}

void op_quant_bilateralslice_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_BILATERALSLICE_COE, dtUINT32, &((op_bilateralslice_t *)op)->coe));
    CHECK(op_setting_single_get(
        op, SETTING_BILATERALSLICE_OFFSET, dtBOOL, &((op_bilateralslice_t *)op)->offset));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_bilateralslice_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_bilateralslice_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_bilateralslice_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_bilateralslice_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_bilateralslice_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_bilateralslice_t *)op)->obits));
}

void op_quant_bilateralslice_tp_destroy(op_t *op)
{
    if (((op_bilateralslice_t *)op)->temp_data)
        mem_delete(((op_bilateralslice_t *)op)->temp_data);
    ((op_bilateralslice_t *)op)->temp_data = NULL;
}

void op_quant_bilateralslice_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}
static void op_quant_bilateralslice_simple_run(op_t *op) // todo : 1. check output
{ // B C H W                                                                            2. guide
    // scale fuse to output
    op_bilateralslice_t *bila_op = (op_bilateralslice_t *)op;
    // size_t batch_size = op->input_tensors[0]->shape.dim[0];
    int i_c = op->input_tensors[0]->shape.dim[1];
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];

    int g_c = op->input_tensors[1]->shape.dim[1];
    int g_h = op->input_tensors[1]->shape.dim[2];
    int g_w = op->input_tensors[1]->shape.dim[3];

    const float scale_x = (float)g_w / i_w;
    const float scale_y = (float)g_h / i_h;

    const uint8_t *input_quant = mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *grid_quant = mem_cpu_data(op->input_tensors[1]->mem);
    const uint8_t *guide_quant = mem_cpu_data(op->input_tensors[2]->mem);
    uint8_t *output = mem_cpu_data(op->output_tensors[0].mem);
    uint32_t *temp_quant = mem_cpu_data(bila_op->temp_data);

    int coe = bila_op->coe;
    int dep = g_c / coe;
    // printf("input:h:%d, w:%d,gc: coe: %d\n", coe);
    bool has_offset = bila_op->offset;

    float ialpha_input = bila_op->ialpha[0];
    float ialpha_grid = bila_op->ialpha[1];
    // float ialpha_guide = bila_op->ialpha[2];

    uint8_t ibits_input = bila_op->ibits[0];
    // uint8_t ibits_grid = bila_op->ibits[1];
    uint8_t ibits_guide = bila_op->ibits[2];
    uint8_t obits_output = bila_op->obits[0];

    uint8_t izero_point_input = bila_op->izero_point[0];
    uint8_t izero_point_grid = bila_op->izero_point[1];
    // uint8_t izero_point_guide = bila_op->izero_point[2];

    float oalpha = bila_op->oalpha[0];
    uint8_t ozero_point = bila_op->ozero_point[0];

    int32_t scale_x_quant = (int32_t)(round(scale_x * (1 << 24)));
    int32_t scale_y_quant = (int32_t)(round(scale_y * (1 << 24)));

    uint32_t input_max = (1 << ibits_input) - 1;
    // uint32_t grid_max = (1 << ibits_grid) - 1;
    uint32_t guide_max = (1 << ibits_guide) - 1;
    uint32_t output_max = (1 << obits_output) - 1;

    int i, j, k, l;
    for (i = 0; i < coe; i++) {
        int depth = dep * i;
        for (j = 0; j < i_h; j++) {
            int32_t gy_quant
                = (((j << 1) + 1) * (int32_t)(scale_y * (1 << 24))); // bit 0 represent 0.5
            int32_t fy_quant = (gy_quant - (1 << 24)) >> 25;
            uint32_t gy_get_quant_dec = ((uint64_t)((j << 9) + 0x100) * scale_y_quant + 1) >> 24;
            uint16_t gy_get_dec = ((gy_get_quant_dec - 0x100) >> 1) & 0xFF;

            for (k = 0; k < i_w; k++) {
                int32_t gx_quant = ((k << 1) + 1) * scale_x_quant; // bit 0 represent 0.5
                int32_t fx_quant = (gx_quant - (1 << 24)) >> 25;
                uint32_t gx_get_quant_dec
                    = ((uint64_t)((k << 9) + 0x100) * scale_x_quant + 1) >> 24;
                uint16_t gx_get_dec = ((gx_get_quant_dec - 0x100) >> 1) & 0xFF;

                int64_t gz_quant
                    = (int64_t)((int64_t)(guide_quant[i_w * j + k]) * (1 << 24) * dep + (1 << 23));
                int32_t fz_quant = ((gz_quant - guide_max * (1 << 23)) >> 24) / guide_max;
                uint32_t gz_get_quant_dec = (((dep << 9)) * guide_quant[i_w * j + k]) / guide_max;
                uint16_t gz_get_dec = ((gz_get_quant_dec - 0x100) >> 1) & 0xFF;

                int x_left = max(min(fx_quant, g_w - 1), 0);
                int x_right = max(min(fx_quant + 1, g_w - 1), 0);
                int y_left = max(min(fy_quant, g_h - 1), 0);
                int y_right = max(min(fy_quant + 1, g_h - 1), 0);
                int z_left = max(min(fz_quant, dep - 1), 0);
                int z_right = max(min(fz_quant + 1, dep - 1), 0);

                int grid_idx_000 = ((z_left + depth) * g_h + y_left) * g_w + x_left;
                int grid_idx_001 = ((z_left + depth) * g_h + y_left) * g_w + x_right;

                int grid_idx_010 = ((z_left + depth) * g_h + y_right) * g_w + x_left;
                int grid_idx_011 = ((z_left + depth) * g_h + y_right) * g_w + x_right;

                int grid_idx_100 = ((z_right + depth) * g_h + y_left) * g_w + x_left;
                int grid_idx_101 = ((z_right + depth) * g_h + y_left) * g_w + x_right;

                int grid_idx_110 = ((z_right + depth) * g_h + y_right) * g_w + x_left;
                int grid_idx_111 = ((z_right + depth) * g_h + y_right) * g_w + x_right;

                int gx_t_int = fx_quant;
                uint8_t gx_t_dec = gx_get_dec;
                uint8_t fabs_res_x = fx_quant - gx_t_int > 0 ? 255 - gx_t_dec : gx_t_dec;
                uint8_t wx_0_quant = 255 - fabs_res_x;
                uint8_t wx_1_quant = fabs_res_x;

                int gy_t_int = fy_quant;
                uint8_t gy_t_dec = gy_get_dec;
                uint8_t fabs_res_y = fy_quant - gy_t_int > 0 ? 255 - gy_t_dec : gy_t_dec;
                uint8_t wy_0_quant = 255 - fabs_res_y;
                uint8_t wy_1_quant = fabs_res_y;

                int gz_t_int = fz_quant;
                uint8_t gz_t_dec = gz_get_dec;
                uint8_t fabs_res_z = fz_quant - gz_t_int > 0 ? 255 - gz_t_dec : gz_t_dec;
                uint8_t wz_0_quant = 255 - fabs_res_z;
                uint8_t wz_1_quant = fabs_res_z;

                uint32_t scale_0 = (wx_0_quant * wy_0_quant * wz_0_quant);
                uint32_t scale_1 = (wx_1_quant * wy_0_quant * wz_0_quant);
                uint32_t scale_2 = (wx_0_quant * wy_1_quant * wz_0_quant);
                uint32_t scale_3 = (wx_1_quant * wy_1_quant * wz_0_quant);

                uint32_t scale_4 = (wx_0_quant * wy_0_quant * wz_1_quant);
                uint32_t scale_5 = (wx_1_quant * wy_0_quant * wz_1_quant);
                uint32_t scale_6 = (wx_0_quant * wy_1_quant * wz_1_quant);
                uint32_t scale_7 = (wx_1_quant * wy_1_quant * wz_1_quant);

                uint32_t val_quant = 0;

                val_quant += (grid_quant[grid_idx_000] * scale_0); // grid
                val_quant += (grid_quant[grid_idx_001] * scale_1);
                val_quant += (grid_quant[grid_idx_010] * scale_2);
                val_quant += (grid_quant[grid_idx_011] * scale_3);
                val_quant += (grid_quant[grid_idx_100] * scale_4);
                val_quant += (grid_quant[grid_idx_101] * scale_5);
                val_quant += (grid_quant[grid_idx_110] * scale_6);
                val_quant += (grid_quant[grid_idx_111] * scale_7);

                temp_quant[(i * i_h + j) * i_w + k] = val_quant;
            }
        }
    }
    for (i = 0; i < i_h; i++) {
        for (j = 0; j < i_w; j++) {
            for (k = 0; k < i_c; k++) {
                int64_t sum_quant_product = 0;
                for (l = 0; l < i_c; l++) {
                    int64_t Qgrid_sub_Zgrid
                        = ((int64_t)(temp_quant[((l * i_c + k) * i_h + i) * i_w + j])
                           - izero_point_grid * 255 * 255 * 255);
                    int16_t Qin_sub_Zin
                        = ((int16_t)(input_quant[(l * i_h + i) * i_w + j]) - izero_point_input);
                    // uint8_t inputimg = input_quant[(l * i_h + i) * i_w + j];
                    sum_quant_product += (Qgrid_sub_Zgrid * Qin_sub_Zin);
                }
                if (has_offset == true) // always be true
                {
                    int64_t Qgrid_sub_Zgrid
                        = ((int64_t)(temp_quant[((k + i_c * i_c) * i_h + i) * i_w + j])
                           - izero_point_grid * 255 * 255 * 255);
                    sum_quant_product += Qgrid_sub_Zgrid * input_max;
                    sum_quant_product = sum_quant_product < 0 ? 0 : sum_quant_product; // clip >0
                }
                uint32_t output_now
                    = (((int64_t)(sum_quant_product * (ialpha_grid * ialpha_input / oalpha * (1 << 24)) + (1 << 23))
                        >> 24)
                       / (255 * 255 * 255))
                    + ozero_point;

                output_now = output_now > output_max ? output_max : output_now;
                output[(k * i_h + i) * i_w + j] = output_now;
            }
        }
    }
}

void op_quant_bilateralslice_tp_prepare(op_t *op)
{
    int i;
    op_bilateralslice_t *bila_op = (op_bilateralslice_t *)op;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    if (NULL == bila_op->temp_data)
        bila_op->temp_data = mem_new(cpu_mem_tp);

    mem_alloc(
        bila_op->temp_data,
        sizeof(uint32_t) * bila_op->coe * op->input_tensors[0]->shape.dim[2]
            * op->input_tensors[0]->shape.dim[3]);

    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        op->run_func = op_quant_bilateralslice_simple_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
