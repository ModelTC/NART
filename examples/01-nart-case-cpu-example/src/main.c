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

#include <assert.h>
#include <stdio.h>

#include "art/default/default_module.h"
#include "art/default/default_op_settings.h"
#include "art/default/default_op_tp.h"
#include "art/op_settings.h"

int main()
{
    workspace_t *ws_cpu = workspace_new(&default_module_tp, NULL);

    op_t *read_output, *write_input;
    op_t *add;
    {
        int shape[] = { 2, 3, 600, 800 };
        read_output = workspace_new_op(ws_cpu, OP_DEFAULT_READ_OUTPUT, 0, NULL);
        CHECK(op_setting_single_set(
            read_output, SETTING_DEFAULT_READ_OUTPUT_PATH, dtSTR, "data/input"));
        CHECK(op_setting_single_set(read_output, SETTING_DEFAULT_READ_OUTPUT_NUM, dtUINT32, 2));
        CHECK(op_setting_single_set(
            read_output, SETTING_DEFAULT_READ_OUTPUT_DTYPE, dtUINT32, dtFLOAT32));
        CHECK(op_setting_single_set(
            read_output, SETTING_DEFAULT_READ_OUTPUT_MEMTP, dtCPTR, workspace_memtype(ws_cpu)));
        op_config(read_output);
        tensor_set_name(&read_output->output_tensors[0], "input1");
        tensor_set_name(&read_output->output_tensors[1], "input2");
        tensor_reshape(&read_output->output_tensors[0], 4, shape);
        tensor_reshape(&read_output->output_tensors[1], 4, shape);
    }
    {
        tensor_t *mv[] = { &read_output->output_tensors[0], &read_output->output_tensors[1] };
        add = workspace_new_op(ws_cpu, OP_ADD, 2, mv);
        op_config(add);
        tensor_set_name(&add->output_tensors[0], "add_out");
    }
    {
        tensor_t *mv[] = { &add->output_tensors[0] };
        write_input = workspace_new_op(ws_cpu, OP_DEFAULT_WRITE_INPUT, 1, mv);
        CHECK(op_setting_single_set(
            write_input, SETTING_DEFAULT_WRITE_INPUT_PATH, dtSTR, "data/output"));
        op_config(write_input);
    }
    op_prepare(read_output);
    op_prepare(add);
    op_prepare(write_input);

    op_run(read_output);
    op_run(add);
    op_run(write_input);

    op_delete(write_input);
    op_delete(add);
    op_delete(read_output);

    workspace_delete(ws_cpu);
    return 0;
}
