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

#ifndef ART_OP_H
#define ART_OP_H

#include <stdint.h>

#include "compat.h"
#include "op_settings.h"
#include "profiler.h"
#include "settings.h"
#include "tensor.h"

#define DECLARE_OP_TP(module, op_name)                                            \
    extern const op_tp_t op_##op_name##_tp;                                       \
    extern struct op_t *op_##module##_##op_name##_tp_alloc(struct workspace_t *); \
    extern void op_##module##_##op_name##_tp_config(struct op_t *);               \
    extern void op_##module##_##op_name##_tp_destroy(op_t *);                     \
    extern void op_##module##_##op_name##_tp_dealloc(op_t *);                     \
    extern void op_##module##_##op_name##_tp_prepare(op_t *);

#define OP_ENTRY(op_tp, module, op_name)                                \
    {                                                                   \
        .tp = &op_tp, .alloc_func = op_##module##_##op_name##_tp_alloc, \
        .config_func = op_##module##_##op_name##_tp_config,             \
        .destroy_func = op_##module##_##op_name##_tp_destroy,           \
        .dealloc_func = op_##module##_##op_name##_tp_dealloc,           \
        .prepare_func = op_##module##_##op_name##_tp_prepare,           \
    }

#if defined(ART_CONSTRAINT_WITH_NAME)
#define OP_SETTING_CONSTRAINT_REQUIRED(iitem, idtype)                                           \
    {                                                                                           \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_REQUIRED, .name = #iitem \
    }

#define OP_SETTING_CONSTRAINT_OPTIONAL(iitem, idtype, dft_value)                              \
    {                                                                                         \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_OPTIONAL,              \
        .constraint.optional.default_value.UVALUE_MEMBER_##idtype = dft_value, .name = #iitem \
    }

#define OP_SETTING_CONSTRAINT_REPEATED(iitem, idtype)                                           \
    {                                                                                           \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_REPEATED, .name = #iitem \
    }

#define OP_SETTING_CONSTRAINT_END()                                                       \
    {                                                                                     \
        .item = SETTING_END, .dtype = dtUNKNOWN, .ctp = ENUM_SETTING_CONSTRAINT_OPTIONAL, \
        .name = NULL                                                                      \
    }
#else
#define OP_SETTING_CONSTRAINT_REQUIRED(iitem, idtype)                           \
    {                                                                           \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_REQUIRED \
    }

#define OP_SETTING_CONSTRAINT_OPTIONAL(iitem, idtype, dft_value)                 \
    {                                                                            \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_OPTIONAL, \
        .constraint.optional.default_value.UVALUE_MEMBER_##idtype = dft_value    \
    }

#define OP_SETTING_CONSTRAINT_REPEATED(iitem, idtype)                           \
    {                                                                           \
        .item = iitem, .dtype = idtype, .ctp = ENUM_SETTING_CONSTRAINT_REPEATED \
    }

#define OP_SETTING_CONSTRAINT_END()                                                      \
    {                                                                                    \
        .item = SETTING_END, .dtype = dtUNKNOWN, .ctp = ENUM_SETTING_CONSTRAINT_OPTIONAL \
    }
#endif

#define DEFINE_OP_TP_STUB(module, op_name, msg)                                               \
    struct op_t *op_##module##_##op_name##_tp_alloc(struct workspace_t *) { LOG_error(msg); } \
    void op_##module##_##op_name##_tp_config(struct op_t *) { LOG_error(msg); }               \
    void op_##module##_##op_name##_tp_destroy(op_t *) { LOG_error(msg); }                     \
    extern void op_##module##_##op_name##_tp_dealloc(op_t *) { LOG_error(msg); }              \
    extern void op_##module##_##op_name##_tp_prepare(op_t *) { LOG_error(msg); }

#ifdef __cplusplus
extern "C" {
#endif

struct op_t;
struct workspace_t;

typedef void (*op_run_func)(struct op_t *);

typedef bool (*op_tp_infer_output_func)(struct op_t *);

typedef struct op_t *(*op_alloc_op_func)(struct workspace_t *);
typedef void (*op_config_op_func)(struct op_t *);
typedef void (*op_destroy_op_func)(struct op_t *);
typedef void (*op_dealloc_op_func)(struct op_t *);
typedef void (*op_prepare_op_func)(struct op_t *);

typedef struct {
    uint64_t op_tp_code;
    const char *name;
    uint16_t min_input_size;
    uint16_t max_input_size;
    uint16_t min_output_size;
    uint16_t max_output_size;
    op_tp_infer_output_func infer_output_func;
    setting_constraint_t constraints[];
} op_tp_t;

typedef struct {
    const op_tp_t *tp;
    op_alloc_op_func alloc_func;
    op_config_op_func config_func;
    op_destroy_op_func destroy_func;
    op_dealloc_op_func dealloc_func;
    op_prepare_op_func prepare_func;
} op_tp_entry_t;

typedef struct op_t {
    op_tp_entry_t *entry;
    struct workspace_t *workspace;
    size_t input_size;
    size_t output_size;
    tensor_t **input_tensors;
    tensor_t *output_tensors;
    op_run_func run_func;
    setting_t setting;
} op_t;

static inline void op_run(op_t *op)
{
//#define PROFILE
#ifdef PROFILE
    struct timespec start, end;
    nart_clocktime(&start);

    op->run_func(op);
    nart_clocktime(&end);
    printf(
        "[%10s]: [%20s] -> [%20s] %.4fms\n", op->entry->tp->name,
        op->input_size >= 1 ? op->input_tensors[0]->name : "NULL",
        op->output_size >= 1 ? op->output_tensors[0].name : "NULL",
        (end.tv_sec - start.tv_sec) * 1000. + (end.tv_nsec - start.tv_nsec) / 1000000.);
#else
    // int h = 0;
    profiler_pointcut_begin("case", op->entry->tp->name);
    op->run_func(op);
    profiler_pointcut_end("case", op->entry->tp->name);
#endif
};

void op_init_default(
    op_t *op, op_tp_entry_t *entry, struct workspace_t *workspace, uint16_t input_size,
    tensor_t *const *input_tensor);

void op_destroy_default(op_t *);

bool op_config(struct op_t *);

static inline bool op_infer_output(op_t *op) { return op->entry->tp->infer_output_func(op); }

void op_prepare(op_t *op);

void op_delete(op_t *op);

bool op_setting_if_set(const op_t *op, uint32_t item);
bool op_setting_single_get(const op_t *op, uint32_t item, uint32_t dtype, void *out);
bool op_setting_single_set(op_t *op, uint32_t item, uint32_t dtype, ...);
bool op_setting_array_set(op_t *op, uint32_t item, uint32_t dtype, size_t len, const void *in);
bool op_setting_array_append(op_t *op, uint32_t item, uint32_t dtype, size_t len, const void *in);
bool op_setting_array_get(const op_t *op, uint32_t item, uint32_t dtype, size_t *len, void *out);

#ifdef __cplusplus
}
#endif

#endif // ART_OP_H
