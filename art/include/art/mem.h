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

#ifndef MEM_H
#define MEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

struct mem_t;

typedef struct mem_t *(*mem_func_new)();
typedef void (*mem_func_delete)(struct mem_t *);

typedef bool (*mem_func_alloc)(struct mem_t *, size_t);
typedef void (*mem_func_free)(struct mem_t *);

typedef void *(*mem_func_data)(const struct mem_t *);

typedef size_t (*mem_func_size)(struct mem_t *);

typedef void *(*mem_func_copy_from_cpu)(void *to, const void *from, size_t sz);
typedef mem_func_copy_from_cpu mem_func_copy_to_cpu;

// function to create mem_t which shares data from given buffer.
typedef struct mem_t *(*mem_func_shared_from)(void *data, size_t size);

typedef struct mem_tp {
    const char *name;
    mem_func_new new_func;
    mem_func_delete delete_func;

    mem_func_alloc alloc_func;
    mem_func_free free_func;

    mem_func_data data_func;
    mem_func_data cpu_data_func;
    mem_func_size size_func;

    mem_func_copy_from_cpu copy_from_cpu_func;
    mem_func_copy_to_cpu copy_to_cpu_func;

    mem_func_shared_from share_from_func;
} mem_tp;

typedef struct mem_t {
    const mem_tp *tp;
    uint32_t refcount;
} mem_t;

static inline mem_t *mem_new(const mem_tp *tp)
{
    mem_t *res = tp->new_func();
    res->refcount = 1;
    res->tp = tp;
    return res;
}

static inline bool mem_alloc(mem_t *m, size_t sz) { return m->tp->alloc_func(m, sz); }

static inline void mem_delete(mem_t *m)
{
    if (NULL != m && 0 == --m->refcount)
        m->tp->delete_func(m);
}

static inline bool mem_tp_check(const mem_t *m, const mem_tp *tp) { return m->tp == tp; }

static inline void *mem_data(const mem_t *m) { return m->tp->data_func(m); }

static inline void *mem_cpu_data(mem_t *m) { return m->tp->cpu_data_func(m); }

static inline size_t mem_sizeof(mem_t *m) { return m->tp->size_func(m); }

static inline mem_t *mem_share_from(const mem_tp *tp, void *data, size_t size)
{
    if (tp->share_from_func == NULL) {
        LOG_error("mem_tp `%s` does not support share from existing buffer\n", tp->name);
    }
    return tp->share_from_func(data, size);
}

extern const mem_tp *cpu_mem_tp;

/* extern mem type */

mem_t *extern_mem_from_data(void *data, size_t sz);

mem_t *mem_dup(mem_t *m);

extern const mem_tp *extern_mem_tp;

#ifdef __cplusplus
}
#endif

#endif // MEM_H
