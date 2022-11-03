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

#ifndef PARADE_H
#define PARADE_H

#include <stdbool.h>

#include "op.h"
#include "transform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _parade_t parade_t;
typedef tensor_t *const *tensor_array_t; /* just like tensor_t*[] */
parade_t *parade_new();
void parade_delete(parade_t *parade);

bool parade_get_input_tensors(parade_t *parade, size_t *count, tensor_array_t *tensors);
bool parade_get_output_tensors(parade_t *parade, size_t *count, tensor_array_t *tensors);

bool parade_get_tensor_producer(parade_t *parade, const char *name, op_t *ops);
bool parade_get_tensor_consumer(parade_t *parade, const char *name, op_t **ops, size_t *len);

bool parade_get_transfrom_param(parade_t *parade, size_t *count, transform_param_t *params);
void parade_set_transfrom_param(parade_t *parade, transform_param_t *params);

void parade_infer_output(parade_t *parade);
void parade_prepare(parade_t *parade);
void parade_run(parade_t *parade);

bool parade_apply_reshape(parade_t *parade);

#ifdef __cplusplus
}
#endif
#endif // PARADE_H
