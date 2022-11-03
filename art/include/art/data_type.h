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

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define dtUNKNOWN 0
/* int */
#define dtINT8  1
#define dtINT16 2
#define dtINT32 3
#define dtINT64 4
/* uint */
#define dtUINT8  5
#define dtUINT16 6
#define dtUINT32 7
#define dtUINT64 8
/* float */
#define dtFLOAT16 9
#define dtFLOAT32 10
#define dtFLOAT64 11

/* other */
#define dtSTR  12
#define dtPTR  13
#define dtCPTR 14
#define dtBOOL 15

#define tp_dtUNKNOWN int8_t

#define tp_dtINT8  int8_t
#define tp_dtINT16 int16_t
#define tp_dtINT32 int32_t
#define tp_dtINT64 int64_t

#define tp_dtUINT8  uint8_t
#define tp_dtUINT16 uint16_t
#define tp_dtUINT32 uint32_t
#define tp_dtUINT64 uint64_t

#define tp_dtFLOAT16 uint16_t
#define tp_dtFLOAT32 float
#define tp_dtFLOAT64 double

#define tp_dtSTR  char *
#define tp_dtPTR  void *
#define tp_dtCPTR const void *
#define tp_dtBOOL bool

#ifdef __cplusplus
extern "C" {
#endif

const char *datatype_name_from_type(uint32_t dtype);
uint32_t datatype_type_from_name(const char *name);
size_t datatype_sizeof(uint32_t dtype);

#ifdef __cplusplus
}
#endif

#define dtSIZEOF(x) SZ_##x

#endif // DATA_TYPE_H
