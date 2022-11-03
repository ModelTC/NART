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

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/quant/quant_mem.h"
#include "art/quant/quant_ops.h"

#include "./op/dequantize.h"
#include "./op/quantize.h"
#include "./quant_workspace.h"

static int quant_ws_num = 0;

static const char *ws_quant_name(const quant_workspace_t *ws) { return ws->name; }

static const mem_tp *ws_quant_mem_tp(quant_workspace_t *ws)
{
    (void)ws;
    return cpu_mem_tp;
}

static void ws_quant_delete(quant_workspace_t *ws)
{
    free(ws->name);
    free(ws);
}

static workspace_t *ws_quant_new(const setting_entry_t *);

static op_tp_entry_t op_entries[] = {
    OP_ENTRY(op_quant_quantize_tp, quant, quantize),
    OP_ENTRY(op_quant_dequantize_tp, quant, dequantize),
    OP_ENTRY(op_conv_2d_tp, quant, conv_2d),
    OP_ENTRY(op_conv_2d_wino_tp, quant, conv_2d_wino),
    OP_ENTRY(op_deconv_2d_tp, quant, deconv_2d),
    OP_ENTRY(op_ip_tp, quant, ip),
    OP_ENTRY(op_relu_tp, quant, relu),
    OP_ENTRY(op_add_tp, quant, add),
    OP_ENTRY(op_eltwise_tp, quant, eltwise),
    OP_ENTRY(op_concat_tp, quant, concat),
    OP_ENTRY(op_pool_tp, quant, pool),
    OP_ENTRY(op_softmax_tp, quant, softmax),
    // OP_ENTRY(op_slice_tp, quant, slice),
    OP_ENTRY(op_interp_tp, quant, interp),
    OP_ENTRY(op_prelu_tp, quant, prelu),
    OP_ENTRY(op_bilateralslice_tp, quant, bilateralslice),
    //{&op_sqrt_tp, op_quant_sqrt_tp_alloc, op_quant_sqrt_tp_init, op_quant_sqrt_tp_destroy,
    // op_quant_sqrt_tp_dealloc, op_quant_sqrt_tp_prepare}, OP_ENTRY(op_add_tp, quant, add),
    //{&op_sub_tp, op_quant_sub_tp_alloc, op_quant_sub_tp_init, op_quant_sub_tp_destroy,
    // op_quant_sub_tp_dealloc, op_quant_sub_tp_prepare},
    //{&op_quant_write_input_tp, op_quant_write_input_tp_alloc, op_quant_write_input_tp_init,
    // op_quant_write_input_tp_destroy, op_quant_write_input_tp_dealloc,
    // op_quant_write_input_tp_prepare},
    //{&op_quant_read_output_tp, op_quant_read_output_tp_alloc, op_quant_read_output_tp_init,
    // op_quant_read_output_tp_destroy, op_quant_read_output_tp_dealloc,
    // op_quant_read_output_tp_prepare},
    //{&op_conv_2d_tp, op_quant_conv_2d_tp_alloc, op_quant_conv_2d_tp_init,
    // op_quant_conv_2d_tp_destroy, op_quant_conv_2d_tp_dealloc, op_quant_conv_2d_tp_prepare},
    //{&op_lrn_tp, op_quant_lrn_tp_alloc, op_quant_lrn_tp_init, op_quant_lrn_tp_destroy,
    // op_quant_lrn_tp_dealloc, op_quant_lrn_tp_prepare},
    //{&op_pool_tp, op_quant_pool_tp_alloc, op_quant_pool_tp_init, op_quant_pool_tp_destroy,
    // op_quant_pool_tp_dealloc, op_quant_pool_tp_prepare},
    //{&op_bn_tp, op_quant_bn_tp_alloc, op_quant_bn_tp_init, op_quant_bn_tp_destroy,
    // op_quant_bn_tp_dealloc, op_quant_bn_tp_prepare},
    //{&op_slice_tp, op_quant_slice_tp_alloc, op_quant_slice_tp_init, op_quant_slice_tp_destroy,
    // op_quant_slice_tp_dealloc, op_quant_slice_tp_prepare},
    //{&op_interp_tp, op_quant_interp_tp_alloc, op_quant_interp_tp_init, op_quant_interp_tp_destroy,
    // op_quant_interp_tp_dealloc, op_quant_interp_tp_prepare},
    { NULL },
};

const module_t DLL_PUBLIC quant_module_tp = {
    .op_group = OP_GROUP_CODE_QUANT,
    .name_func = (ws_name_func)ws_quant_name,
    .memtype_func = (ws_memtp_func)ws_quant_mem_tp,
    .new_func = (ws_new_func)ws_quant_new,
    .delete_func = (ws_delete_func)ws_quant_delete,
    .op_tp_entry = op_entries,
};

workspace_t *ws_quant_new(const setting_entry_t *settings)
{
    (void)settings;
    char tmp[20];
    quant_workspace_t *res = (quant_workspace_t *)malloc(sizeof(quant_workspace_t));
    memset(res, 0, sizeof(*res));
    sprintf(tmp, "quant:%d", quant_ws_num++);
    res->name = malloc(strlen(tmp) + 1);
    strcpy(res->name, tmp);
    res->ws.module_type = &quant_module_tp;
    return (workspace_t *)res;
}
