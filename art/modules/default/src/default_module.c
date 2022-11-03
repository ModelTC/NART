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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/default/default_ops.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"

#include "./op/read_output.h"
#include "./op/write_input.h"

static int default_ws_num = 0;

typedef struct default_workspace_t {
    workspace_t ws;
    char *name;
} default_workspace_t;

static const char *ws_default_name(const default_workspace_t *ws) { return ws->name; }

static const mem_tp *ws_default_mem_tp(default_workspace_t *ws)
{
    (void)ws;
    return cpu_mem_tp;
}

static void ws_default_delete(default_workspace_t *ws)
{
    free(ws->name);
    free(ws);
}

static workspace_t *ws_default_new(const setting_entry_t *);

static op_tp_entry_t op_entries[] = {
    OP_ENTRY(op_sqrt_tp, default, sqrt),
    OP_ENTRY(op_abs_tp, default, abs),
    OP_ENTRY(op_floor_tp, default, floor),
    OP_ENTRY(op_add_tp, default, add),
    OP_ENTRY(op_sub_tp, default, sub),
    OP_ENTRY(op_default_write_input_tp, default, write_input),
    OP_ENTRY(op_default_read_output_tp, default, read_output),
    OP_ENTRY(op_relu_tp, default, relu),
    OP_ENTRY(op_tanh_tp, default, tanh),
    OP_ENTRY(op_prelu_tp, default, prelu),
    OP_ENTRY(op_conv_nd_tp, default, conv_nd),
    OP_ENTRY(op_conv_2d_tp, default, conv_2d),
    OP_ENTRY(op_deform_conv_2d_tp, default, deform_conv_2d),
    OP_ENTRY(op_deconv_2d_tp, default, deconv_2d),
    OP_ENTRY(op_lrn_tp, default, lrn),
    OP_ENTRY(op_pool_tp, default, pool),
    OP_ENTRY(op_ip_tp, default, ip),
    OP_ENTRY(op_bn_tp, default, bn),
    OP_ENTRY(op_batchnorm_tp, default, batchnorm),
    OP_ENTRY(op_concat_tp, default, concat),
    OP_ENTRY(op_slice_tp, default, slice),
    OP_ENTRY(op_interp_tp, default, interp),
    OP_ENTRY(op_sigmoid_tp, default, sigmoid),
    OP_ENTRY(op_softmax_tp, default, softmax),
    OP_ENTRY(op_eltwise_tp, default, eltwise),
    OP_ENTRY(op_reshape_tp, default, reshape),
    OP_ENTRY(op_transpose_tp, default, transpose),
    OP_ENTRY(op_subpixel_tp, default, subpixel),
    OP_ENTRY(op_heatmap2coord_tp, default, heatmap2coord),
    OP_ENTRY(op_exchange_tp, default, exchange),
    OP_ENTRY(op_roipooling_tp, default, roipooling),
    OP_ENTRY(op_roialignpooling_tp, default, roialignpooling),
    OP_ENTRY(op_psroipooling_tp, default, psroipooling),
    OP_ENTRY(op_correlation_tp, default, correlation),
    OP_ENTRY(op_psroimaskpooling_tp, default, psroimaskpooling),
    OP_ENTRY(op_bilateralslice_tp, default, bilateralslice),
    OP_ENTRY(op_relu6_tp, default, relu6),
    OP_ENTRY(op_scale_tp, default, scale),
    OP_ENTRY(op_shufflechannel_tp, default, shufflechannel),
    OP_ENTRY(op_psroialignpooling_tp, default, psroialignpooling),
    OP_ENTRY(op_pad_tp, default, pad),
    OP_ENTRY(op_quant_dequant_tp, default, quant_dequant),
    OP_ENTRY(op_podroialignpooling_tp, default, podroialignpooling),
    OP_ENTRY(op_instancenorm_tp, default, instancenorm),
    OP_ENTRY(op_mul_tp, default, mul),
    OP_ENTRY(op_div_tp, default, div),
    OP_ENTRY(op_pow_tp, default, pow),
    OP_ENTRY(op_log_tp, default, log),
    OP_ENTRY(op_exp_tp, default, exp),
    OP_ENTRY(op_matmul_tp, default, matmul),
    OP_ENTRY(op_reducemin_tp, default, reducemin),
    OP_ENTRY(op_reducemax_tp, default, reducemax),
    OP_ENTRY(op_reducemean_tp, default, reducemean),
    OP_ENTRY(op_reduceprod_tp, default, reduceprod),
    OP_ENTRY(op_reducesum_tp, default, reducesum),
    OP_ENTRY(op_reducel2_tp, default, reducel2),
    OP_ENTRY(op_podroialignpooling_tp, default, podroialignpooling),
    OP_ENTRY(op_correlation1d_tp, default, correlation1d),
    OP_ENTRY(op_lpnormalization_tp, default, lpnormalization),
    OP_ENTRY(op_gather_tp, default, gather),
    OP_ENTRY(op_argmax_tp, default, argmax),
    OP_ENTRY(op_gridsample_tp, default, gridsample),
    OP_ENTRY(op_unfold_tp, default, unfold),
    OP_ENTRY(op_topk_tp, default, topk),
    OP_ENTRY(op_lstm_tp, default, lstm),
    OP_ENTRY(op_hardsigmoid_tp, default, hardsigmoid),
    OP_ENTRY(op_erf_tp, default, erf),
    OP_ENTRY(op_clip_tp, default, clip),
    OP_ENTRY(op_cast_tp, default, cast),
    OP_ENTRY(op_hswish_tp, default, hswish),
    OP_ENTRY(op_scatternd_tp, default, scatternd),
    OP_ENTRY(op_min_tp, default, min),
    OP_ENTRY(op_sign_tp, default, sign),
    OP_ENTRY(op_roundto0_tp, default, roundto0),
    OP_ENTRY(op_elu_tp, default, elu),
    OP_ENTRY(op_clip_cast_tp, default, clip_cast),
    OP_ENTRY(op_add_div_clip_cast_tp, default, add_div_clip_cast),
    { NULL },
};

const module_t DLL_PUBLIC default_module_tp = {
    .op_group = OP_GROUP_CODE_DEFAULT,
    .name_func = (ws_name_func)ws_default_name,
    .memtype_func = (ws_memtp_func)ws_default_mem_tp,
    .new_func = (ws_new_func)ws_default_new,
    .delete_func = (ws_delete_func)ws_default_delete,
    .op_tp_entry = op_entries,
};

workspace_t *ws_default_new(const setting_entry_t *settings)
{
    (void)settings;
    char tmp[20];
    default_workspace_t *res = (default_workspace_t *)malloc(sizeof(default_workspace_t));
    memset(res, 0, sizeof(*res));
    sprintf(tmp, "default:%d", default_ws_num++);
    res->name = malloc(strlen(tmp) + 1);
    strcpy(res->name, tmp);
    res->ws.module_type = &default_module_tp;
    return (workspace_t *)res;
}
