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

#include <inttypes.h>
#include <signal.h>

#include "art/log.h"
#include "art/module.h"

op_t *workspace_new_op(
    workspace_t *ws, uint64_t op_tp_code, uint8_t input_size, tensor_t *const *input_tensor)
{
    int i;
    op_tp_entry_t *entry;
    CHECK(ws != NULL);
    const module_t *module = ws->module_type;
    if (NULL == module->op_tp_entry)
        return NULL;

    for (i = 0; i < input_size; ++i) {
        if (NULL == input_tensor[i]) {
            DLOG(error, "empty tensor not supported\n");
        }
    }
    for (entry = module->op_tp_entry; NULL != entry && NULL != entry->tp; ++entry) {
        if (op_tp_code == entry->tp->op_tp_code) {
            op_t *op = entry->alloc_func(ws);
            if (NULL == op)
                return NULL;
            // DLOG(info, "found op type '%s' in workspace '%s'\n", entry->tp->name,
            // workspace_name(ws));
            op_init_default(op, entry, ws, input_size, input_tensor);
            return op;
        }
    }
    LOG(error, "Op typecode: 0x%" PRIx64 " not found\n", op_tp_code);
    return NULL;
}

bool workspace_support_op(workspace_t *ws, uint64_t op_tp_code, bool strict)
{
    const module_t *module = ws->module_type;
    op_tp_entry_t *entry;
    if (NULL == module->op_tp_entry)
        return false;
    if (op_tp_code & 0x80000000l)
        strict = true;
    if (strict) {
        for (entry = module->op_tp_entry; NULL != entry && NULL != entry->tp; ++entry) {
            if (op_tp_code == (entry->tp->op_tp_code | ((uint64_t)module->op_group << 32))) {
                return true;
            }
        }
    } else {
        for (entry = module->op_tp_entry; NULL != entry && NULL != entry->tp; ++entry) {
            if ((op_tp_code & 0xffffffffl) == (entry->tp->op_tp_code & 0xffffffffl)) {
                return true;
            }
        }
    }
    return false;
}
