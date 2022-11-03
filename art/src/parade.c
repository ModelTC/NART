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

#include "parade_impl.h"

parade_t *parade_new()
{
    parade_t *res = (parade_t *)malloc(sizeof(parade_t));
    memset(res, 0, sizeof(struct _parade_t));
    return res;
}

void parade_delete(parade_t *parade)
{
    size_t i;
    for (i = 0; i < parade->op_count; ++i) {
        op_delete(parade->ops[i]);
    }
    free(parade->ops);
    for (i = 0; i < parade->input_tensor_count; ++i) {
        tensor_delete(parade->input_tensors[i]);
    }
    for (i = 0; i < parade->weight_tensor_count; ++i) {
        tensor_delete(parade->weight_tensors[i]);
    }
    if (parade->input_tensor_count > 0)
        free(parade->input_tensors);
    if (parade->output_tensor_count > 0)
        free(parade->output_tensors);
    if (parade->weight_tensor_count > 0)
        free(parade->weight_tensors);
    free(parade);
}

bool parade_get_input_tensors(parade_t *parade, size_t *count, tensor_array_t *tensors)
{
    if (NULL == parade)
        return false;
    size_t input_count = parade->input_tensor_count;
    if (NULL != count)
        *count = input_count;
    // if (0 == input_count)
    //     return false;
    if (NULL != tensors)
        *tensors = parade->input_tensors;
    return true;
}

bool parade_get_output_tensors(parade_t *parade, size_t *count, tensor_array_t *tensors)
{
    if (NULL == parade)
        return false;
    size_t output_count = parade->output_tensor_count;
    if (NULL != count)
        *count = output_count;
    // if (0 == output_count)
    //     return false;
    if (NULL != tensors)
        *tensors = parade->output_tensors;
    return true;
}

bool parade_get_tensor_producer(parade_t *parade, const char *name, op_t *ops)
{
    if (NULL == parade || NULL == name)
        return false;
    size_t i, j;
    for (i = 0; i < parade->op_count; i++) {
        op_t *op = parade->ops[i];
        for (j = 0; j < op->output_size; j++) {
            if (op->output_tensors[j].name != NULL
                && 0 == strcmp(name, op->output_tensors[j].name)) {
                *ops = *op;
                return true;
            }
        }
    }
    return false;
}

bool parade_get_tensor_consumer(parade_t *parade, const char *name, op_t **ops, size_t *len)
{
    if (NULL == parade)
        return false;
    size_t i, j, cnt = 0;
    for (i = 0; i < parade->op_count; i++) {
        op_t *op = parade->ops[i];
        for (j = 0; j < op->input_size; j++) {
            if (op->input_tensors[j]->name != NULL
                && 0 == strcmp(name, op->input_tensors[j]->name)) {
                cnt++;
            }
        }
    }
    if (cnt == 0)
        return false;

    // need to `free` by caller
    *ops = malloc(sizeof(op_t) * cnt);
    *len = cnt;
    cnt = 0;

    for (i = 0; i < parade->op_count; i++) {
        op_t *op = parade->ops[i];
        for (j = 0; j < op->input_size; j++) {
            if (op->input_tensors[j]->name != NULL
                && 0 == strcmp(name, op->input_tensors[j]->name)) {
                (*ops)[cnt++] = *op;
            }
        }
    }
    return true;
}

bool parade_get_transfrom_param(parade_t *parade, size_t *count, transform_param_t *params)
{
    if (NULL == parade)
        return false;
    size_t input_count = parade->input_tensor_count;
    if (NULL != count)
        *count = input_count;
    if (NULL != params)
        *params = ((transform_tensor_t *)(parade->input_tensors[0]))->param;
    return true;
}

void parade_set_transfrom_param(parade_t *parade, transform_param_t *params)
{
    int i;
    for (i = 0; i < parade->input_tensor_count; i++) {
        if (!strncmp(params->tensor_name, parade->input_tensors[i]->name, 64)) {
            if (parade->input_tensors[i]->with_transform) {
                transform_tensor_t *t = (transform_tensor_t *)parade->input_tensors[i];
                t->param = *params;
                break;
            }
            printf("tensor [%s] found, but with no transform flag\n", params->tensor_name);
        }
    }
}

void parade_infer_output(parade_t *parade)
{
    size_t i;
    if (NULL == parade)
        return;
    const size_t count = parade->op_count;
    if (0 == count)
        return;
    op_t **ops = parade->ops;
    for (i = 0; i < count; ++i)
        op_infer_output(ops[i]);
    return;
}

void parade_prepare(parade_t *parade)
{
    size_t i;
    if (NULL == parade)
        return;
    const size_t count = parade->op_count;
    if (0 == count)
        return;
    op_t **ops = parade->ops;
    for (i = 0; i < count; ++i) {
        op_prepare(ops[i]);
    }
    return;
}

void parade_run(parade_t *parade)
{
    size_t i;
    if (NULL == parade)
        return;
    const size_t count = parade->op_count;
    if (0 == count)
        return;
    op_t **ops = parade->ops;
    for (i = 0; i < count; ++i) {
        op_run(ops[i]);
    }
    return;
}

bool parade_apply_reshape(parade_t *parade)
{
    size_t i;
    if (NULL == parade)
        return false;
    const size_t count = parade->op_count;
    if (0 == count)
        return true;
    op_t **ops = parade->ops;
    for (i = 0; i < count; ++i) {
        if (ops[i]->entry->tp->infer_output_func(ops[i]) == false) {
            return false;
        }
        ops[i]->entry->prepare_func(ops[i]);
    }
    return true;
}
