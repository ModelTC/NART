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

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "art/default/default_op_settings.h"
#include "art/default/default_op_tp.h"
#include "art/default/default_ops.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/settings_helper.h"

typedef struct {
    op_t o;
    char *input_path;
    uint32_t dtype;
    const mem_tp *memtp;
} op_read_output_t;

static bool op_infer_shape_default_read_output(op_t *op)
{
    int i;
    CHECK_EQ(0, op->input_size);
    uint32_t sz_output = 0;
    const mem_tp *mem_tp = NULL;
    CHECK(op_setting_single_get(op, SETTING_DEFAULT_READ_OUTPUT_NUM, dtUINT32, &sz_output));
    if (false == op_setting_if_set(op, SETTING_DEFAULT_READ_OUTPUT_MEMTP)) {
        mem_tp = workspace_memtype(op->workspace);
    } else {
        CHECK(op_setting_single_get(op, SETTING_DEFAULT_READ_OUTPUT_MEMTP, dtCPTR, &mem_tp));
    }
    op->output_size = sz_output;
    op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
    memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
    for (i = 0; i < op->output_size; ++i) {
        op->output_tensors[i].mem = mem_new(mem_tp);
        op->output_tensors[i].dtype = ((op_read_output_t *)op)->dtype;
        op->output_tensors[i].shape.dim_size = 4;
        op->output_tensors[i].shape.batch_axis = 0;
#ifdef __leon__
        op->output_tensors[i].shape.channel_axis = 3;
#else
        op->output_tensors[i].shape.channel_axis = 1;
#endif
    }

    return true;
}

const op_tp_t op_default_read_output_tp = {
    .op_tp_code = OP_DEFAULT_READ_OUTPUT,
    .name = "read_output",
    .min_input_size = 0,
    .max_input_size = 0,
    .min_output_size = 1,
    .max_output_size = 0xffff,
    .infer_output_func = op_infer_shape_default_read_output,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_DEFAULT_READ_OUTPUT_PATH, dtSTR),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_DEFAULT_READ_OUTPUT_NUM, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_DEFAULT_READ_OUTPUT_DTYPE, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_DEFAULT_READ_OUTPUT_MEMTP, dtCPTR, &cpu_mem_tp),
       OP_SETTING_CONSTRAINT_END()}
};

op_t *op_default_read_output_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_read_output_t *res = (op_read_output_t *)malloc(sizeof(op_read_output_t));
    memset(res, 0, sizeof(*res));
    return (op_t *)res;
}

void op_default_read_output_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_DEFAULT_READ_OUTPUT_MEMTP, dtCPTR, &((op_read_output_t *)op)->memtp));
    CHECK(op_setting_single_get(
        op, SETTING_DEFAULT_READ_OUTPUT_PATH, dtSTR, &((op_read_output_t *)op)->input_path));
    CHECK(op_setting_single_get(
        op, SETTING_DEFAULT_READ_OUTPUT_DTYPE, dtUINT32, &((op_read_output_t *)op)->dtype));
}

void op_default_read_output_tp_destroy(op_t *op)
{
    if (NULL != ((op_read_output_t *)op)->input_path) {
        // free(((op_read_output_t*)op)->input_path);
    }
}

void op_default_read_output_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_read_output_run(op_t *op)
{
    int i;
    char tmp[128];
    for (i = 0; i < op->output_size; ++i) {
        sprintf(
            tmp, "%s/%s.bin", ((op_read_output_t *)op)->input_path,
            tensor_name(&op->output_tensors[i]));
        FILE *f = fopen(tmp, "r");
        if (NULL == f) {
            LOG(warn, "open file [%s] failed\n", tmp);
            continue;
        }
        CHECK_EQ(
            1,
            fread(
                mem_cpu_data(op->output_tensors[i].mem),
                datatype_sizeof(op->output_tensors[i].dtype)
                    * shape_count(&op->output_tensors[i].shape),
                1, f));
        CHECK_EQ(0, fclose(f));
    }
}

void op_default_read_output_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->output_size; ++i) {
        CHECK_NE(NULL, tensor_name(&op->output_tensors[i]));
        tensor_alloc(&op->output_tensors[i]);
    }
    op->run_func = op_default_read_output_run;
}
