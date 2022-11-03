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

#ifndef MODULE_H
#define MODULE_H

#include "op.h"
#include "settings.h"
#include "workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WIN32
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllexport))
#else
#define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL  __attribute__((visibility("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif

typedef const char *(*ws_name_func)(const workspace_t *);
typedef const mem_tp *(*ws_memtp_func)(const workspace_t *);

typedef workspace_t *(*ws_new_func)(const setting_entry_t *);
typedef void (*ws_delete_func)(workspace_t *);

typedef struct _module_t {
    uint32_t op_group;
    ws_name_func name_func;
    ws_memtp_func memtype_func;

    ws_new_func new_func;
    ws_delete_func delete_func;

    op_tp_entry_t *op_tp_entry;
} module_t;

static inline workspace_t *workspace_new(const module_t *module, const setting_entry_t *settings)
{
    return module->new_func(settings);
}

static inline void workspace_delete(workspace_t *ws) { ws->module_type->delete_func(ws); }

static inline const char *workspace_name(workspace_t *ws) { return ws->module_type->name_func(ws); }

static inline const mem_tp *workspace_memtype(workspace_t *ws)
{
    return ws->module_type->memtype_func(ws);
}

op_t *workspace_new_op(
    workspace_t *ws, uint64_t op_tp_code, uint8_t input_size, tensor_t *const *input_tensor);

bool workspace_support_op(workspace_t *ws, uint64_t op_tp_code, bool strict);

#ifdef __cplusplus
}
#endif
#endif // MODULE_H
