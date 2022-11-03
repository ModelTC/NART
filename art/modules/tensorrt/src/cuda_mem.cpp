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

#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/mem.h"
#include "art/tensorrt/cuda_mem.h"

#include "cuda_runtime.h"

typedef struct cuda_mem_t {
    mem_t m;
    size_t sz;
    void *data;
    void *cpu_data;
    bool gpu_updated;
} cuda_mem_t;

static mem_t *cuda_mem_new()
{
    cuda_mem_t *res = NULL;
    res = (cuda_mem_t *)malloc(sizeof(cuda_mem_t));
    memset((void *)res, 0, sizeof(cuda_mem_t));
    ((cuda_mem_t *)res)->gpu_updated = false;
    return (mem_t *)res;
}

static void cuda_mem_free(cuda_mem_t *m);
static void cuda_mem_delete(cuda_mem_t *m)
{
    cuda_mem_free(m);
    free(m);
}

static bool cuda_mem_alloc(cuda_mem_t *m, size_t sz)
{
    if (m->sz >= sz)
        return true;
    if (0 < m->sz) {
        CUDA_CHECK(cudaFree(m->data));
    }
    void *data = NULL;
    CUDA_CHECK(cudaMalloc(&data, sz));
    m->data = data;
    m->sz = sz;
    data = realloc(m->cpu_data, sz);
    CHECK_NE(NULL, data);
    m->cpu_data = data;
    return true;
}

static void cuda_mem_free(cuda_mem_t *m)
{
    if (m->sz > 0) {
        CUDA_CHECK(cudaFree(m->data));
        free(m->cpu_data);
        m->sz = 0;
        m->data = NULL;
    }
}

static void *cuda_mem_data(cuda_mem_t *m)
{
    if (false == m->gpu_updated) {
        CUDA_CHECK(cudaMemcpy(m->data, m->cpu_data, m->sz, cudaMemcpyHostToDevice));
        m->gpu_updated = true;
    }
    return m->data;
}

static void *cuda_mem_cpu_data(cuda_mem_t *m)
{
    if (true == m->gpu_updated) {
        CUDA_CHECK(cudaMemcpy(m->cpu_data, m->data, m->sz, cudaMemcpyDeviceToHost));
        m->gpu_updated = false;
    }
    return m->cpu_data;
}

static size_t cuda_mem_size(cuda_mem_t *m) { return m->sz; }

static void *cuda_mem_cpy_to_cpu(void *dst, const void *from, size_t sz)
{
    CUDA_CHECK(cudaMemcpy(dst, from, sz, cudaMemcpyDeviceToHost));
    return dst;
}

static void *cuda_mem_cpy_from_cpu(void *dst, const void *from, size_t sz)
{
    CUDA_CHECK(cudaMemcpy(dst, from, sz, cudaMemcpyHostToDevice));
    return dst;
}

// create an shared_cuda_mem which shares data from buffer dev_data.
mem_t *cuda_mem_share_from(void *dev_data, size_t size);

static const mem_tp cuda_mem_tp_st = [] {
    mem_tp mt;
    mt.name = "cuda_mem";
    mt.new_func = cuda_mem_new;
    mt.alloc_func = (mem_func_alloc)cuda_mem_alloc;
    mt.free_func = (mem_func_free)cuda_mem_free;
    mt.delete_func = (mem_func_delete)cuda_mem_delete;
    mt.data_func = (mem_func_data)cuda_mem_data;
    mt.cpu_data_func = (mem_func_data)cuda_mem_cpu_data;
    mt.size_func = (mem_func_size)cuda_mem_size;
    mt.copy_to_cpu_func = cuda_mem_cpy_to_cpu;
    mt.copy_from_cpu_func = cuda_mem_cpy_from_cpu;
    mt.share_from_func = cuda_mem_share_from;
    return mt;
}();

extern "C" const mem_tp *const trt_mem_tp = &cuda_mem_tp_st;

/* #### shared cuda mem #### */
static bool shared_cuda_mem_alloc(cuda_mem_t *m, size_t sz)
{
    if (m->sz >= sz)
        return true;
    LOG_warn("shared_cuda_mem doesn't support allocating more space\n");
    return false;
}

static void shared_cuda_mem_free(cuda_mem_t *m)
{
    if (m->sz > 0) {
        free(m->cpu_data);
        m->data = NULL;
        m->sz = 0;
    }
}

static void shared_cuda_mem_delete(cuda_mem_t *m)
{
    shared_cuda_mem_free(m);
    free(m);
}

static const mem_tp shared_cuda_mem_tp_st = [] {
    mem_tp mt;
    mt.name = "shared_cuda_mem";
    mt.new_func = nullptr;
    // the alloc function is difference, to disable allocating.
    mt.alloc_func = (mem_func_alloc)shared_cuda_mem_alloc;
    mt.free_func = (mem_func_free)shared_cuda_mem_free;
    mt.delete_func = (mem_func_delete)shared_cuda_mem_delete;
    mt.data_func = (mem_func_data)cuda_mem_data;
    mt.cpu_data_func = (mem_func_data)cuda_mem_cpu_data;
    mt.size_func = (mem_func_size)cuda_mem_size;
    mt.copy_to_cpu_func = cuda_mem_cpy_to_cpu;
    mt.copy_from_cpu_func = cuda_mem_cpy_from_cpu;
    mt.share_from_func = cuda_mem_share_from;
    return mt;
}();

mem_t *cuda_mem_share_from(void *dev_data, size_t size)
{
    cuda_mem_t *ret = static_cast<cuda_mem_t *>(malloc(sizeof(cuda_mem_t)));
    memset(ret, 0, sizeof(*ret));
    ret->m.tp = &shared_cuda_mem_tp_st;
    ret->gpu_updated = true;
    ret->data = dev_data;
    ret->cpu_data = malloc(size);
    ret->sz = size;
    ret->m.refcount = 1;
    return reinterpret_cast<mem_t *>(ret);
}

/*
int trt_mem_set_gpu(mem_t* mem, int gpuid)
{
if (NULL == mem || mem->tp != trt_mem_tp) {
return -1;
}
if (gpuid == ((cuda_mem_t*)mem)->gpu)
return gpuid;
CUDA_CHECK(cudaSetDevice(((cuda_mem_t*)mem)->gpu));
size_t sz = ((cuda_mem_t*)mem)->sz;
if (sz) {
CUDA_CHECK(cudaFree(((cuda_mem_t*)mem)->data));
}
CUDA_CHECK(cudaSetDevice(gpuid));
CUDA_CHECK(cudaMalloc(&((cuda_mem_t*)mem)->data, sz));
((cuda_mem_t*)mem)->gpu = gpuid;
return gpuid;
}

int trt_mem_get_gpu(mem_t* mem)
{
if (NULL == mem || mem->tp != trt_mem_tp) {
return -1;
}
return ((cuda_mem_t*)mem)->gpu;
}
*/
