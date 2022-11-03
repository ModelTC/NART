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

#include <string.h>

#define ST_dtUNKNOWN ptr

#define ST_dtINT8  i8
#define ST_dtINT16 i16
#define ST_dtINT32 i32
#define ST_dtINT64 i64

#define ST_dtUINT8  u8
#define ST_dtUINT16 u16
#define ST_dtUINT32 u32
#define ST_dtUINT64 u64

#define ST_dtFLOAT16 f16
#define ST_dtFLOAT32 f32
#define ST_dtFLOAT64 f64

#define ST_dtSTR  str
#define ST_dtPTR  ptr
#define ST_dtCPTR cptr
#define ST_dtBOOL b

#define SETTING_COPY_VALUE(to, from) \
    do {                             \
        to = from;                   \
    } while (0)

#define SETTING_COPY_ST_dtINT8  SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtINT16 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtINT32 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtINT64 SETTING_COPY_VALUE

#define SETTING_COPY_ST_dtUINT8  SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtUINT16 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtUINT32 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtUINT64 SETTING_COPY_VALUE

#define SETTING_COPY_ST_dtFLOAT16 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtFLOAT32 SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtFLOAT64 SETTING_COPY_VALUE

#define SETTING_COPY_ST_dtBOOL SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtPTR  SETTING_COPY_VALUE
#define SETTING_COPY_ST_dtCPTR SETTING_COPY_VALUE

#define SETTING_COPY_ST_dtSTR(to, from)       \
    do {                                      \
        char *tmp = malloc(strlen(from) + 1); \
        strcpy(tmp, from);                    \
        if (NULL != to) {                     \
            free(to);                         \
            to = NULL;                        \
        }                                     \
        to = tmp;                             \
    } while (0)

#define SETTING_COPY(to, from, type) SETTING_COPY_##type(to, from)
