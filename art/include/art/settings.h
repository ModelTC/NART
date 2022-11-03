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

#ifndef SETTINGS_H
#define SETTINGS_H

#include "data_type.h"
#include "settings_helper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;

    float f32;
    double f64;

    char *str;

    const void *cptr;
    void *ptr;

    int i;
    unsigned int u;
    bool b;
} uvalue_t;

typedef enum {
    ENUM_SETTING_VALUE_SINGLE = 1,
    ENUM_SETTING_VALUE_REPEATED = 2,
} enum_setting_value_tp_t;

typedef enum {
    ENUM_SETTING_CONSTRAINT_REQUIRED = 1,
    ENUM_SETTING_CONSTRAINT_OPTIONAL = 2,
    ENUM_SETTING_CONSTRAINT_REPEATED = 3,
} enum_setting_constraint_tp_t;

typedef struct {
    uint32_t item;
    uint32_t dtype;
    enum_setting_constraint_tp_t ctp;
    union {
        struct {
            int _;
        } required;
        struct {
            uvalue_t default_value;
        } optional;
        struct {
            int _;
        } repeated;
    } constraint;
#if defined(ART_CONSTRAINT_WITH_NAME)
    const char *name;
#endif
} setting_constraint_t;

typedef struct single_setting_value_t {
    uvalue_t value;
} single_setting_value_t;

typedef struct repeated_setting_value_t {
    void *values;
    size_t len;
} repeated_setting_value_t;

typedef struct setting_entry_t {
    uint32_t item;
    uint32_t dtype;
    enum_setting_value_tp_t tp;
    union {
        single_setting_value_t single;
        repeated_setting_value_t repeated;
    } v;
} setting_entry_t;

typedef struct _setting_t {
    size_t len;
    size_t reserved;
    setting_entry_t entries[];
} * setting_t;

setting_t setting_new();
setting_t setting_new_with_reserved(size_t count);
void setting_delete(setting_t setting);
void setting_shrink(setting_t *setting);
setting_entry_t *setting_search(setting_t setting, uint32_t item);

void setting_set_single(setting_t *setting, uint32_t item, uint32_t dtype, uvalue_t v);
void *setting_alloc_repeated(setting_t *setting, uint32_t item, uint32_t dtype, size_t count);

#ifdef __cplusplus
}
#endif

#define UVALUE_MEMBER(dtype) UVALUE_MEMBER_##dtype

#define UVALUE_MEMBER_dtINT8    i8
#define UVALUE_MEMBER_dtINT16   i16
#define UVALUE_MEMBER_dtINT32   i32
#define UVALUE_MEMBER_dtINT64   i64
#define UVALUE_MEMBER_dtUINT8   u8
#define UVALUE_MEMBER_dtUINT16  u16
#define UVALUE_MEMBER_dtUINT32  u32
#define UVALUE_MEMBER_dtUINT64  u64
#define UVALUE_MEMBER_dtFLOAT32 f32
#define UVALUE_MEMBER_dtFLOAT16 f16
#define UVALUE_MEMBER_dtSTR     str
#define UVALUE_MEMBER_dtPTR     ptr
#define UVALUE_MEMBER_dtCPTR    cptr
#define UVALUE_MEMBER_dtBOOL    b

#define SETTING_ENTRY_AS_TYPE(dt_type, entry) SETTING_ENTRY_AS_##dt_type(entry)

#define SETTING_ENTRY_AS_dtINT8(entry) (CHECK_EQ(dtINT8, entry->dtype), entry->value.i8)

#define SETTING_ENTRY_AS_dtINT16(entry) (CHECK_EQ(dtINT16, entry->dtype), entry->value.i16)

#define SETTING_ENTRY_AS_dtINT32(entry) (CHECK_EQ(dtINT32, entry->dtype), entry->value.i32)

#define SETTING_ENTRY_AS_dtINT64(entry) (CHECK_EQ(dtINT64, entry->dtype), entry->value.i64)

#define SETTING_ENTRY_AS_dtUINT8(entry) (CHECK_EQ(dtUINT8, entry->dtype), entry->value.u8)

#define SETTING_ENTRY_AS_dtUINT16(entry) (CHECK_EQ(dtUINT16, entry->dtype), entry->value.u16)

#define SETTING_ENTRY_AS_dtUINT32(entry) (CHECK_EQ(dtUINT32, entry->dtype), entry->value.u32)

#define SETTING_ENTRY_AS_dtUINT64(entry) (CHECK_EQ(dtUINT64, entry->dtype), entry->value.u64)

#define SETTING_ENTRY_AS_dtFLOAT16(entry) (CHECK_EQ(dtFLOAT16, entry->dtype), entry->value.f16)

#define SETTING_ENTRY_AS_dtFLOAT32(entry) (CHECK_EQ(dtFLOAT32, entry->dtype), entry->value.f32)

#define SETTING_ENTRY_AS_dtFLOAT64(entry) (CHECK_EQ(dtFLOAT64, entry->dtype), entry->value.f64)

#define SETTING_ENTRY_AS_dtSTR(entry) \
    (CHECK_EQ(dtSTR, entry->dtype), (const char *)entry->value.str)

#define SETTING_ENTRY_AS_dtPTR(entry) (CHECK_EQ(dtPTR, entry->dtype), entry->value.ptr)

#define SETTING_ENTRY_AS_dtCPTR(entry) \
    (CHECK(dtPTR == entry->dtype || dtCPTR == entry->dtype), entry->value.cptr)

#define SETTING_ENTRY_AS_dtBOOL(entry) (CHECK_EQ(dtBOOL, entry->dtype), entry->value.b)

#define SETTING_ENTRY_AS_ARRAY(dt_type, entry)                                      \
    (CHECK_EQ(ENUM_SETTING_REPEATED, entry->tp), (CHECK_EQ(dt_type, entry->dtype)), \
     ((tp_##dt_type *)entry->v.repeated.values))
#endif // SETTINGS_H
