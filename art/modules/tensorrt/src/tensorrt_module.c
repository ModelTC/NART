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

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/tensorrt/tensorrt_ops.h"
#include "art/tensorrt/tensorrt_workspace.h"

//#include "art/op_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/tensorrt/cuda_mem.h"
#include "art/tensorrt/tensorrt_op_settings.h"
#include "art/tensorrt/tensorrt_op_tp.h"
#include "art/tensorrt/tensorrt_ws_settings.h"

static int tensorrt_ws_num = 0;

extern bool op_infer_shape_tensorrt_net(op_t *op);
const op_tp_t op_tensorrt_net_tp = {
    .op_tp_code = OP_TENSORRT_NET,
    .name = "tensorrt_net",
    .min_input_size = 1,
    .max_input_size = 0xffff,
    .min_output_size = 1,
    .max_output_size = 0xffff,
    .infer_output_func = op_infer_shape_tensorrt_net,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_TENSORRT_NET, dtUINT8),
                    OP_SETTING_CONSTRAINT_REPEATED(SETTING_TENSORRT_OUTPUTS, dtSTR),
                    OP_SETTING_CONSTRAINT_REPEATED(SETTING_TENSORRT_INPUTS, dtSTR),
                    OP_SETTING_CONSTRAINT_END()}
};

static const char *ws_tensorrt_name(const tensorrt_workspace_t *ws) { return ws->name; }

static const mem_tp *ws_tensorrt_mem_tp(tensorrt_workspace_t *ws)
{
    (void)ws;
    return trt_mem_tp; // todo check here! should be gpu?
}

static void ws_tensorrt_delete(tensorrt_workspace_t *ws)
{
    // CUDA_CHECK(cudaStreamDestroy(ws->stream));
    free(ws->name);
    free(ws);
}

static workspace_t *ws_tensorrt_new(const setting_entry_t *);

static op_tp_entry_t op_entries[] = {
    OP_ENTRY(op_tensorrt_net_tp, tensorrt, net),
    { NULL },
};

const module_t DLL_PUBLIC tensorrt_module_tp = {
    .op_group = OP_GROUP_CODE_TENSORRT,
    .name_func = (ws_name_func)ws_tensorrt_name,
    .memtype_func = (ws_memtp_func)ws_tensorrt_mem_tp,
    .new_func = (ws_new_func)ws_tensorrt_new,
    .delete_func = (ws_delete_func)ws_tensorrt_delete,
    .op_tp_entry = op_entries,
};

workspace_t *ws_tensorrt_new(const setting_entry_t *settings)
{
    uint32_t dev_id = 0;
    if (settings != NULL) {
        const setting_entry_t *iter = settings;
        for (; iter->item != 0; ++iter) {
            switch (iter->item) {
            case SETTING_TENSORRT_WORKSPACE_DEVID:
                CHECK_EQ(iter->dtype, dtUINT32);
                CHECK_EQ(iter->tp, ENUM_SETTING_VALUE_SINGLE);
                dev_id = iter->v.single.value.u32;
                break;
            default:
                CHECK(true);
            }
        }
    }

    char tmp[20];
    tensorrt_workspace_t *res = (tensorrt_workspace_t *)malloc(sizeof(tensorrt_workspace_t));
    memset(res, 0, sizeof(*res));

    sprintf(tmp, "tensorrt:%d", tensorrt_ws_num++);
    res->name = malloc(strlen(tmp) + 1);
    strcpy(res->name, tmp);
    res->ws.module_type = &tensorrt_module_tp;

    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    stream = 0;
    res->stream = stream;
    res->dev_id = dev_id;
    return (workspace_t *)res;
}
