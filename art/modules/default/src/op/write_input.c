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

static bool op_infer_shape_default_write_input(op_t *op)
{
    CHECK_LE(1, op->input_size);
    if (0 == op->output_size) {
        op->output_size = op->input_size;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        int i;
        for (i = 0; i < op->output_size; ++i) {
            op->output_tensors[i] = *op->input_tensors[i];
            op->output_tensors[i].mem->refcount += 1;
            if (NULL != op->output_tensors[i].name) {
                char *name = (char *)malloc(strlen(op->output_tensors[i].name) + 1);
                strcpy(name, op->output_tensors[i].name);
                DCHECK_NE(NULL, name);
                op->output_tensors[i].name = name;
            }
        }
    }
    return true;
}

const op_tp_t op_default_write_input_tp = {
    .op_tp_code = OP_DEFAULT_WRITE_INPUT,
    .name = "write_input",
    .min_input_size = 1,
    .max_input_size = 0xffff,
    .min_output_size = 1,
    .max_output_size = 0xffff,
    .infer_output_func = op_infer_shape_default_write_input,
    .constraints = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_DEFAULT_WRITE_INPUT_PATH, dtSTR),
                    OP_SETTING_CONSTRAINT_END()}
};

typedef struct {
    op_t o;
    const char *output_path;
} op_write_input_t;

op_t *op_default_write_input_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_write_input_t *res = (op_write_input_t *)malloc(sizeof(op_write_input_t));
    memset(res, 0, sizeof(*res));
    return (op_t *)res;
}

void op_default_write_input_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_DEFAULT_WRITE_INPUT_PATH, dtSTR, &((op_write_input_t *)op)->output_path));
}

void op_default_write_input_tp_destroy(op_t *op)
{
    (void)op;
    DCHECK_NE(NULL, op);
}

void op_default_write_input_tp_dealloc(op_t *op)
{
    if (NULL != op) {
        free(op);
    }
}

static void op_default_write_input_run(op_t *op)
{
    int i;
    char tmp[128];
    for (i = 0; i < op->input_size; ++i) {
        sprintf(
            tmp, "%s/%s.bin", ((op_write_input_t *)op)->output_path,
            tensor_name(op->input_tensors[i]));
        FILE *f = fopen(tmp, "w+");
        if (NULL == f) {
            LOG(warn, "open file [%s] failed\n", tmp);
            continue;
        }
        CHECK_EQ(
            1,
            fwrite(
                mem_cpu_data(op->input_tensors[i]->mem),
                datatype_sizeof(op->input_tensors[i]->dtype)
                    * shape_count(&op->input_tensors[i]->shape),
                1, f));
        CHECK_EQ(0, fclose(f));
    }
}

void op_default_write_input_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        CHECK_NE(NULL, tensor_name(op->input_tensors[i]));
        tensor_alloc(op->input_tensors[i]);
    }
    op->run_func = op_default_write_input_run;
}
