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
#include <stdlib.h>
#include <string.h>

#include "art/data_type.h"

#define dtype_count (sizeof(dtype_names) / sizeof(const char *))
static const char *dtype_names[] = {
    "UNKNOWN", "INT8",    "INT16",   "INT32",  "INT64",

    "UINT8",   "UINT16",  "UINT32",  "UINT64",

    "FLOAT16", "FLOAT32", "FLOAT64",

    "STR",     "PTR",     "CPTR",    "BOOL",
};

const char *datatype_name_from_type(uint32_t dtype)
{
    if (dtype >= dtype_count)
        return NULL;
    else
        return dtype_names[dtype];
}

uint32_t datatype_type_from_name(const char *name)
{
    uint32_t i;
    if (NULL == name)
        return 0;
    for (i = 0; i < dtype_count; ++i) {
        if (0 == strncmp(name, dtype_names[i], strlen(dtype_names[i])))
            return i;
    }
    return 0;
}

size_t datatype_sizeof(uint32_t dtype)
{
    switch (dtype) {
    case dtINT8:
    case dtUINT8:
        return sizeof(int8_t);
    case dtINT16:
    case dtUINT16:
    case dtFLOAT16:
        return sizeof(int16_t);
    case dtINT32:
    case dtUINT32:
    case dtFLOAT32:
        return sizeof(int32_t);
    case dtINT64:
    case dtUINT64:
    case dtFLOAT64:
        return sizeof(int64_t);
    case dtSTR:
    case dtPTR:
        return sizeof(void *);
    case dtBOOL:
        return sizeof(bool);
    default:
        return 0;
    }
}
