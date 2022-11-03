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

#include <cuda_module.h>
#include <cuda_op_settings.h>
#include <cuda_op_tp.h>
#include <module.h>
#include <op_settings.h>
#include <op_tp.h>
#include <settings.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
    workspace_t *ws = workspace_new(&default_module_tp, NULL);
    const mem_tp *memtype = workspace_memtype(ws);
    (void)memtype;

    op_t *read_output, *write_input;
    op_t *sqrt, *relu, *add, *deform;

    {
        int shape[] = { 1, 1, 3 };
        read_output = workspace_new_op(ws, OP_DEFAULT_READ_OUTPUT, 0, NULL);
        op_setting_single_set(read_output, SETTING_DEFAULT_READ_OUTPUT_PATH, dtSTR, "input");
        op_setting_single_set(read_output, SETTING_DEFAULT_READ_OUTPUT_NUM, dtUINT32, 2);
        op_setting_single_set(read_output, SETTING_DEFAULT_READ_OUTPUT_DTYPE, dtUINT32, dtFLOAT32);
        op_config(read_output);
        tensor_set_name(&read_output->output_tensors[0], "input1");
        tensor_set_name(&read_output->output_tensors[1], "input2");
        tensor_reshape(&read_output->output_tensors[0], 3, shape);
        tensor_reshape(&read_output->output_tensors[1], 3, shape);
    }
    {
        tensor_t *mv[] = { &read_output->output_tensors[0] };
        relu = workspace_new_op(ws, OP_RELU, 1, mv);
        op_config(relu);
        tensor_set_name(&relu->output_tensors[0], "relu_out");
    }
    {
        tensor_t *mv[] = { &relu->output_tensors[0] };
        sqrt = workspace_new_op(ws, OP_SQRT, 1, mv);
        op_config(sqrt);
        tensor_set_name(&sqrt->output_tensors[0], "sqrt_out");
    }
    {
        tensor_t *mv[] = { &sqrt->output_tensors[0], &read_output->output_tensors[1] };
        add = workspace_new_op(ws, OP_ADD, 2, mv);
        op_config(add);
        tensor_set_name(&add->output_tensors[0], "add_out");
    }
    {
        tensor_t *mv[] = { &sqrt->output_tensors[0], &read_output->output_tensors[1],
                           &sqrt->output_tensors[0] };
        deform = workspace_new_op(ws, OP_DEFORM_CONV_2D, 3, mv);
        op_config(deform);
        tensor_set_name(&deform->output_tensors[0], "deform_out");
    }
    {
        tensor_t *mv[] = {
            &read_output->output_tensors[0], &read_output->output_tensors[1],
            &relu->output_tensors[0],        &sqrt->output_tensors[0],
            &add->output_tensors[0],         &deform->output_tensors[0],
        };
        write_input = workspace_new_op(ws, OP_DEFAULT_WRITE_INPUT, 6, mv);
        op_setting_single_set(write_input, SETTING_DEFAULT_READ_OUTPUT_PATH, dtSTR, "output");
        op_config(write_input);
    }

    op_prepare(read_output);
    op_prepare(relu);
    op_prepare(sqrt);
    op_prepare(add);
    op_prepare(deform);
    op_prepare(write_input);

    op_run(read_output);
    op_run(relu);
    op_run(sqrt);
    op_run(add);
    op_run(deform);
    op_run(write_input);

    op_delete(write_input);
    op_delete(add);
    op_delete(deform);
    op_delete(sqrt);
    op_delete(relu);
    op_delete(read_output);

    workspace_delete(ws);
    return 0;
}
