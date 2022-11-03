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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/mem.h"

typedef struct cpu_mem_t {
    mem_t m;
    size_t sz;
    void *data;
} cpu_mem_t;

static mem_t *cpu_mem_new()
{
    cpu_mem_t *res = NULL;
    res = (cpu_mem_t *)malloc(sizeof(cpu_mem_t));
    memset((void *)res, 0, sizeof(cpu_mem_t));
    return (mem_t *)res;
}

static void cpu_mem_free(cpu_mem_t *m);
static void cpu_mem_delete(cpu_mem_t *m)
{
    cpu_mem_free(m);
    free(m);
}

static bool cpu_mem_alloc(cpu_mem_t *m, size_t sz)
{
    if (m->sz >= sz)
        return true;
    void *data = realloc(m->data, sz);
    if (data == NULL)
        return false;
    m->data = data;
    m->sz = sz;
    return true;
}

static void cpu_mem_free(cpu_mem_t *m)
{
    if (m->sz > 0) {
        free(m->data);
        m->data = NULL;
        m->sz = 0;
    }
}

static void *cpu_mem_data(cpu_mem_t *m) { return m->data; }

static size_t cpu_mem_size(cpu_mem_t *m) { return m->sz; }

static const mem_tp cpu_mem_tp_st = {
    .name = "cpu_mem",
    .new_func = cpu_mem_new,
    .alloc_func = (mem_func_alloc)cpu_mem_alloc,
    .free_func = (mem_func_free)cpu_mem_free,
    .delete_func = (mem_func_delete)cpu_mem_delete,
    .data_func = (mem_func_data)cpu_mem_data,
    .cpu_data_func = (mem_func_data)cpu_mem_data,
    .size_func = (mem_func_size)cpu_mem_size,
    .copy_to_cpu_func = memcpy,
    .copy_from_cpu_func = memcpy,
    .share_from_func = extern_mem_from_data,
};

const mem_tp *cpu_mem_tp = &cpu_mem_tp_st;

/* extern mem type */
typedef cpu_mem_t extern_mem_t;

static const mem_tp extern_mem_tp_st = {
    .name = "cpu_mem",
    .new_func = NULL,
    .alloc_func = NULL,
    .free_func = NULL,
    .delete_func = (mem_func_delete)free,
    .data_func = (mem_func_data)cpu_mem_data,
    .cpu_data_func = (mem_func_data)cpu_mem_data,
    .size_func = (mem_func_size)cpu_mem_size,
    .copy_to_cpu_func = NULL,
    .copy_from_cpu_func = NULL,
};

mem_t *extern_mem_from_data(void *data, size_t sz)
{
    extern_mem_t *res = (extern_mem_t *)malloc(sizeof(extern_mem_t));
    memset(res, 0, sizeof(*res));
    res->m.tp = &extern_mem_tp_st;
    res->data = data;
    res->sz = sz;
    res->m.refcount++;
    return (mem_t *)res;
}

mem_t *mem_dup(mem_t *m)
{
    if (NULL == m)
        return NULL;
    m->refcount++;
    return m;
}

const mem_tp *extern_mem_tp = &extern_mem_tp_st;
