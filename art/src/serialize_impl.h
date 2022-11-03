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

/* serialize */

#include <stdint.h>

#include "art/serialize.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct parade_serializer_t parade_serializer_t;
typedef struct parade_deserializer_t parade_deserializer_t;

typedef struct {
    buffer_t *buffer;
} parade_serializer_tp_t;

typedef struct {
    void (*deserialize_head_func)(parade_deserializer_t *, parade_t *);
    buffer_t *buffer;
} parade_deserializer_tp_t;

struct parade_serializer_t {
    const parade_serializer_tp_t *tp;
    buffer_t *buffer;
};

int32_t deserialize_read_int32(buffer_t *buf);
int64_t deserialize_read_int64(buffer_t *buf);

#ifdef __cplusplus
}
#endif
