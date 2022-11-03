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

#include <stdint.h>
#include <string.h>

#include "art/log.h"
#include "art/serialize.h"

#include "./parade_impl.h"
#include "./serialize_impl.h"

/* buffer */
struct buffer_t {
    size_t pos;
    size_t len;
    nart_buffer_read_func read_func;
    void *param_read;
    nart_buffer_write_func write_func;
    void *param_write;
    size_t buffer_size;
    uint8_t data[];
};

buffer_t *nart_buffer_new(size_t buffer_size)
{
    size_t sz = buffer_size + sizeof(buffer_t);
    buffer_t *res = (buffer_t *)malloc(sz);
    if (NULL == res)
        return NULL;
    memset(res, 0, sz);
    res->buffer_size = buffer_size;
    return res;
}

void nart_buffer_delete(buffer_t *buf)
{
    if (NULL != buf)
        free(buf);
}

void *buffer_data(buffer_t *buf)
{
    if (NULL == buf)
        return NULL;
    return buf->data;
}

size_t nart_buffer_read(buffer_t *buf, size_t sz, void *data)
{
    if (NULL == buf || sz == 0 || NULL == data)
        return 0;
    size_t rem_size = sz;
    while (rem_size > 0) {
        if (buf->pos == buf->len) {
            if (NULL == buf->read_func)
                return sz - rem_size;
            buf->len = buf->read_func(buf->buffer_size, buf->data, buf->param_read);
            buf->pos = 0;
            if (0 == buf->len)
                return sz - rem_size;
        }
        size_t read_size = rem_size;
        if (buf->pos + read_size > buf->len)
            read_size = buf->len - buf->pos;
        memcpy(data, buf->data + buf->pos, read_size);
        buf->pos += read_size;
        rem_size -= read_size;
        data = ((uint8_t *)data) + read_size;
    }
    return sz - rem_size;
}

size_t nart_buffer_write(buffer_t *buf, size_t sz, const void *data)
{
    if (NULL == buf || sz == 0 || NULL == data)
        return 0;
    size_t rem_size = sz;
    while (rem_size > 0) {
        if (buf->len == buf->buffer_size) {
            if (NULL == buf->write_func)
                return sz - rem_size;
            if (buf->len != buf->write_func(buf->len, buf->data, buf->param_write)) {
                LOG(error, "failed write buffer!\n");
            }
            buf->pos = 0;
            buf->len = 0;
            if (0 == rem_size)
                return sz - rem_size;
        }
        size_t write_size = rem_size;
        if (buf->len + write_size > buf->buffer_size)
            write_size = buf->buffer_size - buf->len;
        memcpy(buf->data + buf->len, data, write_size);
        buf->len += write_size;
        rem_size -= write_size;
        data = ((uint8_t *)data) + write_size;
    }
    return sz - rem_size;
}

void buffer_flush_write(buffer_t *buf)
{
    if (NULL == buf || NULL == buf->write_func)
        return;
    if (buf->len != buf->write_func(buf->len, buf->data, buf->param_write)) {
        LOG(error, "failed flush write buffer!\n");
    }
    buf->pos = 0;
}

void buffer_set_buffer_read_func(buffer_t *buf, nart_buffer_read_func func, void *param)
{
    if (NULL == buf)
        return;
    buf->read_func = func;
    buf->param_read = param;
}

void buffer_set_buffer_write_func(buffer_t *buf, nart_buffer_write_func func, void *param)
{
    if (NULL == buf)
        return;
    buf->write_func = func;
    buf->param_write = param;
}

/* end of buffer */

void serialize_write_int32(buffer_t *buf, int32_t value)
{
    nart_buffer_write(buf, sizeof(int32_t), &value);
}

int32_t deserialize_read_int32(buffer_t *buf)
{
    int32_t res = 0;
    CHECK_EQ(sizeof(int32_t), nart_buffer_read(buf, sizeof(int32_t), &res));
    return res;
}

int64_t deserialize_read_int64(buffer_t *buf)
{
    int64_t res = 0;
    CHECK_EQ(sizeof(int64_t), nart_buffer_read(buf, sizeof(int64_t), &res));
    return res;
}

void serialize_write_string(buffer_t *buf, const char *string)
{
    size_t len = NULL == string ? 0 : strlen(string);
    serialize_write_int32(buf, len);
    if (0 == len)
        return;
    char *tmp = (char *)malloc(len);
    memcpy(tmp, string, len);
    nart_buffer_write(buf, len, tmp);
    free(tmp);
}

char *deserialize_read_string(buffer_t *buf)
{
    size_t len = deserialize_read_int32(buf);
    if (0 == len)
        return NULL;
    char *res = (char *)malloc(len + 1);
    CHECK_NE(NULL, res);
    CHECK_NE(0, nart_buffer_read(buf, len, res));
    res[len] = 0;
    return res;
}

serialize_head_t *deserialize_read_head(buffer_t *buf)
{
    serialize_head_t *res = (serialize_head_t *)malloc(sizeof(serialize_head_t));
    memset(res, 0, sizeof(serialize_head_t));
    res->version = deserialize_read_int32(buf);
    res->name = deserialize_read_string(buf);
    return res;
}

void serialize_write_head(buffer_t *buf, const serialize_head_t *head)
{
    serialize_write_int32(buf, head->version);
    serialize_write_string(buf, head->name);
}

void serialize_head_free(serialize_head_t *head)
{
    if (NULL == head)
        return;
    if (NULL != head->name)
        free(head->name);
    free(head);
}

extern parade_t *deserialize_parade_v1(buffer_t *buf, const deserialize_param_t *param);
extern parade_t *deserialize_parade_v2(buffer_t *buf, const deserialize_param_t *param);
parade_t *deserialize_parade(buffer_t *buf, const deserialize_param_t *param)
{
    serialize_head_t *head = deserialize_read_head(buf);
    parade_t *res = NULL;
    switch (head->version) {
    case 1:
        res = deserialize_parade_v1(buf, param);
        break;
    case 2:
        res = deserialize_parade_v2(buf, param);
        break;
    default:
        CHECK(false);
    }
    serialize_head_free(head);
    return res;
}

shape_t deserialize_read_shape(buffer_t *buf)
{
    shape_t res;
    uint32_t dim_size = deserialize_read_int32(buf);
    CHECK_GE(MAX_DIM, dim_size);
    res.dim_size = dim_size;
    nart_buffer_read(buf, dim_size * sizeof(uint32_t), &res.dim);
    nart_buffer_read(buf, sizeof(int32_t), &res.channel_axis);
    nart_buffer_read(buf, sizeof(int32_t), &res.batch_axis);
    return res;
}

void serialize_write_shape(buffer_t *buf, const shape_t *shape)
{
    serialize_write_int32(buf, MAX_DIM);
    nart_buffer_write(buf, sizeof(shape_t), shape);
}

void serialize_write_setting_single(buffer_t *buf, uint32_t item, uint32_t dtype, const void *data)
{
    serialize_write_int32(buf, false);
    serialize_write_int32(buf, item);
    serialize_write_int32(buf, dtype);
    switch (dtype) {
    case dtUINT8:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtINT8:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtUINT16:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtINT16:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtUINT32:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtINT32:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtBOOL:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtFLOAT16:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtFLOAT32:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtFLOAT64:
        nart_buffer_write(buf, datatype_sizeof(dtype), data);
        break;
    case dtSTR:
        serialize_write_string(buf, data);
        break;
    default:
        CHECK(false);
    }
}

pixel_t deserialize_read_pixel(buffer_t *buf)
{
    pixel_t res;
    nart_buffer_read(buf, sizeof(float), &res.r);
    nart_buffer_read(buf, sizeof(float), &res.g);
    nart_buffer_read(buf, sizeof(float), &res.b);
    return res;
}

void serialize_write_pixel(buffer_t *buf, const pixel_t *pixel)
{
    nart_buffer_write(buf, sizeof(pixel_t), pixel);
}
