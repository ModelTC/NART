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

#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <stdint.h>
#include <stdlib.h>

#include "module.h"
#include "parade.h"

#ifdef __cplusplus
extern "C" {
#endif

/* buffer */
typedef size_t (*nart_buffer_write_func)(size_t sz, const void *data, void *param);
typedef size_t (*nart_buffer_read_func)(size_t sz, void *data, void *param);

typedef struct buffer_t buffer_t;

buffer_t *nart_buffer_new(size_t buffer_size);
void nart_buffer_delete(buffer_t *buf);

void *buffer_data(buffer_t *buf);

size_t nart_buffer_read(buffer_t *buf, size_t sz, void *data);
size_t nart_buffer_write(buffer_t *buf, size_t sz, const void *data);

void buffer_set_buffer_read_func(buffer_t *buf, nart_buffer_read_func func, void *param);
void buffer_set_buffer_write_func(buffer_t *buf, nart_buffer_write_func func, void *param);

void buffer_flush_write(buffer_t *buf);

void serialize_write_int32(buffer_t *buf, int32_t value);
void serialize_write_string(buffer_t *buf, const char *string);

typedef struct {
    uint32_t version;
    char *name;
} serialize_head_t;

typedef struct {
    workspace_t *const *workspaces;
    const mem_tp *input_mem_tp;
} deserialize_param_t;

serialize_head_t *deserialize_read_head(buffer_t *buf);
parade_t *deserialize_parade(buffer_t *buf, const deserialize_param_t *param);

void serialize_write_head(buffer_t *buf, const serialize_head_t *head);

void serialize_head_free(serialize_head_t *head);

char *deserialize_read_string(buffer_t *buf);

void serialize_write_shape(buffer_t *buf, const shape_t *shape);
shape_t deserialize_read_shape(buffer_t *buf);

void serialize_write_setting_single(buffer_t *buf, uint32_t item, uint32_t type, const void *data);

void serialize_write_pixel(buffer_t *buf, const pixel_t *pixel);
pixel_t deserialize_read_pixel(buffer_t *buf);

#ifdef __cplusplus
}
#endif

#endif // SERIALIZE_H
