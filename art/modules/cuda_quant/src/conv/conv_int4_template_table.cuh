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

#ifndef CONV_INT4_TEMPLATE_TABLE_CUH
#define CONV_INT4_TEMPLATE_TABLE_CUH

#include "art/cuda_quant/conv/turing_indirect_conv_help.cuh"
#include "art/cuda_quant/conv/int4/turing_indirect_conv_wmma_int4_ldg16_K32.cuh"
#include "art/cuda_quant/conv/int4/turing_indirect_conv_wmma_int4_ldg16_K64.cuh"
#include "art/cuda_quant/conv/int4/turing_indirect_conv_wmma_int4_ldg16_K128.cuh"
#include "art/cuda_quant/conv/int4/turing_indirect_conv_wmma_int4_ldg16_K256.cuh"

#include <vector>
#include <functional>



extern std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> >int4_conv_K32_table;

extern std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *,workspace_t*)> >int4_conv_K64_table;


extern std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> >int4_conv_K128_table;

extern std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *,workspace_t*)> >int4_conv_K256_table;


std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K32_table =
    {
        // thread(4, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,128,32,16,16,32,4,1, true>,
        // lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,64,32,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,256,32,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,256,32,16,16,32,4,1, true>,
        // thread(2, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,128,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,64,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,256,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,64,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,256,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,128,32,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,64,32,16,16,32,2,1, true>,
        // thread(1, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,128,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,64,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,256,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,64,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,256,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,128,32,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<64,64,32,16,16,32,1,2, true>,
        // thread(2, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,256,32,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,128,32,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,256,32,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,2,2, true>,
        // thread(1, 4)
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,256,32,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,128,32,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,256,32,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,1,4, true>,

#ifdef NO_OVERLAP
        lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,128,32,16,16,32,4,1,false>,
        // lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<256,64,32,16,16,32,4,1, false>,
        // lb_turing_indirect_conv_wmma_int4_ldg16_k32_singlebuf_relu<128,64,32,16,16,32,4,1, false>,
#endif
    };


std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K64_table =
    {
        // thread(4, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<256,64,64,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,128,64,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,256,64,16,16,32,4,1, true>,
        // thread(2, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,32,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,128,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,32,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,128,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,64,64,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,32,64,16,16,32,2,1, true>,
        // thread(1, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,32,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,128,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,32,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,128,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,64,64,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<32,32,64,16,16,32,1,2, true>,
        // thread(2, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<256,64,64,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,256,64,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,128,64,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,2,2, true>,
        // thread(1, 4)
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<256,64,64,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,256,64,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,128,64,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,1,4, true>,

#ifdef NO_OVERLAP
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,128,64,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<256,64,64,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<128,64,64,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_singlebuf_relu<64,64,64,16,16,32,4,1, false>,
#endif
    };


std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K128_table =
    {
        // matrixB_singlebuf
        // thread(4, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,128,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<256,64,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,64,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,64,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<256,32,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,32,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,128,128,16,16,32,4,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,32,128,16,16,32,4,1>,
        // thread(2, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,128,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,64,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<128,32,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,128,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,64,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<64,32,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<32,128,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<32,64,128,16,16,32,2,1>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_matrixB_singlebuf_relu<32,32,128,16,16,32,2,1>,


        // singlebuf
        // thread(4, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<256,64,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<256,32,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,32,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,128,128,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,32,128,16,16,32,4,1, true>,
        // thread(2, 1)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,32,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,128,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,32,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,128,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,64,128,16,16,32,2,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,32,128,16,16,32,2,1, true>,
        // thread(1, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,32,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,128,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,32,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,128,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,64,128,16,16,32,1,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,32,128,16,16,32,1,2, true>,
        // thread(2, 2)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,32,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,128,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,32,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,128,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,64,128,16,16,32,2,2, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,32,128,16,16,32,2,2, true>,
        // thread(1, 4)
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,256,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,128,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,256,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,128,128,16,16,32,1,4, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<32,64,128,16,16,32,1,4, true>,

#ifdef NO_OVERLAP
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,128,128,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<256,64,128,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<128,64,128,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_singlebuf_relu<64,64,128,16,16,32,4,1, false>,
#endif

    };

std::vector<std::function<
    void(char*, char*, char*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K256_table =
    {

        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<128,128,256,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<128,64,256,16,16,32,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<64,64,256,16,16,32,4,1, true>,
#ifdef NO_OVERLAP
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<128,128,256,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<128,64,256,16,16,32,4,1, false>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_singlebuf_relu<64,64,256,16,16,32,4,1, false>,
#endif

    };

extern std::vector<std::vector<int> > int4_conv_K32_table_tiling;
extern std::vector<std::vector<int> > int4_conv_K64_table_tiling;
extern std::vector<std::vector<int> > int4_conv_K128_table_tiling;
extern std::vector<std::vector<int> > int4_conv_K256_table_tiling;

std::vector<std::vector<int>> int4_conv_K32_table_tiling = {

        // thread(4, 1)
        {128,128,32,16,16,32, 4,1},
        {256,128,32,16,16,32, 4,1},
        // {128,64,32,16,16,32, 4,1},
        {128,256,32,16,16,32, 4,1},
        {256,256,32,16,16,32, 4,1},
        // thread(2, 1)
        {256,128,32,16,16,32, 2,1},
        {256,64,32,16,16,32, 2,1},
        {128,256,32,16,16,32, 2,1},
        {128,128,32,16,16,32, 2,1},
        {128,64,32,16,16,32, 2,1},
        {64,256,32,16,16,32, 2,1},
        {64,128,32,16,16,32, 2,1},
        {64,64,32,16,16,32, 2,1},
        // thread(1, 2)
        {256,128,32,16,16,32, 1,2},
        {256,64,32,16,16,32, 1,2},
        {128,256,32,16,16,32, 1,2},
        {128,128,32,16,16,32, 1,2},
        {128,64,32,16,16,32, 1,2},
        {64,256,32,16,16,32, 1,2},
        {64,128,32,16,16,32, 1,2},
        {64,64,32,16,16,32, 1,2},
        // thread(2, 2)
        {256,256,32,16,16,32, 2,2},
        {256,128,32,16,16,32, 2,2},
        {128,256,32,16,16,32, 2,2},
        {128,128,32,16,16,32, 2,2},
        // thread(1, 4)
        {256,256,32,16,16,32, 1,4},
        {256,128,32,16,16,32, 1,4},
        {128,256,32,16,16,32, 1,4},
        {256,256,32,16,16,32, 1,4},

#ifdef NO_OVERLAP
        {128,128,32,16,16,32, 4,1},
        // {256,64,32,16,16,32, 4,1},
        // {128,64,32,16,16,32, 4,1},
#endif
    };

std::vector<std::vector<int>> int4_conv_K64_table_tiling = {
    // thread(4, 1)
    {128,128,64,16,16,32, 4,1},
    {256,64,64,16,16,32, 4,1},
    {128,64,64,16,16,32, 4,1},
    {64,64,64,16,16,32, 4,1},
    {64,128,64,16,16,32, 4,1},
    {64,256,64,16,16,32, 4,1},
    // thread(2, 1)
    {128,128,64,16,16,32, 2,1},
    {128,64,64,16,16,32, 2,1},
    {128,32,64,16,16,32, 2,1},
    {64,128,64,16,16,32, 2,1},
    {64,64,64,16,16,32, 2,1},
    {64,32,64,16,16,32, 2,1},
    {32,128,64,16,16,32, 2,1},
    {32,64,64,16,16,32, 2,1},
    {32,32,64,16,16,32, 2,1},
    // thread(1, 2)
    {128,128,64,16,16,32, 1,2},
    {128,64,64,16,16,32, 1,2},
    {128,32,64,16,16,32, 1,2},
    {64,128,64,16,16,32, 1,2},
    {64,64,64,16,16,32, 1,2},
    {64,32,64,16,16,32, 1,2},
    {32,128,64,16,16,32, 1,2},
    {32,64,64,16,16,32, 1,2},
    {32,32,64,16,16,32, 1,2},
    // thread(2, 2)
    {256,64,64,16,16,32, 2,2},
    {128,128,64,16,16,32, 2,2},
    {128,64,64,16,16,32, 2,2},
    {64,256,64,16,16,32, 2,2},
    {64,128,64,16,16,32, 2,2},
    {64,64,64,16,16,32, 2,2},
    // thread(1, 4)
    {256,64,64,16,16,32, 1,4},
    {128,128,64,16,16,32, 1,4},
    {128,64,64,16,16,32, 1,4},
    {64,256,64,16,16,32, 1,4},
    {64,128,64,16,16,32, 1,4},
    {64,64,64,16,16,32, 1,4},

#ifdef NO_OVERLAP
    {128,128,64,16,16,32, 4,1},
    {256,64,64,16,16,32, 4,1},
    {128,64,64,16,16,32, 4,1},
    {64,64,64,16,16,32, 4,1},
#endif
};

std::vector<std::vector<int>> int4_conv_K128_table_tiling = {
        // matrixB_singlebuf
        // thread(4, 1)
        {128,128,128,16,16,32, 4,1},
        {256,64,128,16,16,32, 4,1},
        {128,64,128,16,16,32, 4,1},
        {64,64,128,16,16,32, 4,1},
        {256,32,128,16,16,32, 4,1},
        {128,32,128,16,16,32, 4,1},
        {64,128,128,16,16,32, 4,1},
        {64,32,128,16,16,32, 4,1},
        // thread(2, 1)
        {128,128,128,16,16,32, 2,1},
        {128,64,128,16,16,32, 2,1},
        {128,32,128,16,16,32, 2,1},
        {64,128,128,16,16,32, 2,1},
        {64,64,128,16,16,32, 2,1},
        {64,32,128,16,16,32, 2,1},
        {32,128,128,16,16,32, 2,1},
        {32,64,128,16,16,32, 2,1},
        {32,32,128,16,16,32, 2,1},

        // singlebuf
        // thread(4, 1)
        {128,128,128,16,16,32, 4,1},
        {256,64,128,16,16,32, 4,1},
        {128,64,128,16,16,32, 4,1},
        {64,64,128,16,16,32, 4,1},
        {256,32,128,16,16,32, 4,1},
        {128,32,128,16,16,32, 4,1},
        {64,128,128,16,16,32, 4,1},
        {64,32,128,16,16,32, 4,1},
        // thread(2, 1)
        {128,128,128,16,16,32, 2,1},
        {128,64,128,16,16,32, 2,1},
        {128,32,128,16,16,32, 2,1},
        {64,128,128,16,16,32, 2,1},
        {64,64,128,16,16,32, 2,1},
        {64,32,128,16,16,32, 2,1},
        {32,128,128,16,16,32, 2,1},
        {32,64,128,16,16,32, 2,1},
        {32,32,128,16,16,32, 2,1},
        // thread(1, 2)
        {128,128,128,16,16,32, 1,2},
        {128,64,128,16,16,32, 1,2},
        {128,32,128,16,16,32, 1,2},
        {64,128,128,16,16,32, 1,2},
        {64,64,128,16,16,32, 1,2},
        {64,32,128,16,16,32, 1,2},
        {32,128,128,16,16,32, 1,2},
        {32,64,128,16,16,32, 1,2},
        {32,32,128,16,16,32, 1,2},
        // thread(2, 2)
        {128,128,128,16,16,32, 2,2},
        {128,64,128,16,16,32, 2,2},
        {128,32,128,16,16,32, 2,2},
        {64,128,128,16,16,32, 2,2},
        {64,64,128,16,16,32, 2,2},
        {64,32,128,16,16,32, 2,2},
        {32,128,128,16,16,32, 2,2},
        {32,64,128,16,16,32, 2,2},
        {32,32,128,16,16,32, 2,2},
        // thread(1, 4)
        {128,128,128,16,16,32, 1,4},
        {128,64,128,16,16,32, 1,4},
        {64,256,128,16,16,32, 1,4},
        {64,128,128,16,16,32, 1,4},
        {64,64,128,16,16,32, 1,4},
        {32,256,128,16,16,32, 1,4},
        {32,128,128,16,16,32, 1,4},
        {32,64,128,16,16,32, 1,4},


#ifdef NO_OVERLAP
        {128,128,128,16,16,32, 4,1},
        {256,64,128,16,16,32, 4,1},
        {128,64,128,16,16,32, 4,1},
        {64,64,128,16,16,32, 4,1},
#endif
    };

std::vector<std::vector<int>> int4_conv_K256_table_tiling = {

        {128,128,256,16,16,32, 4,1},
        {128,64,256,16,16,32, 4,1},
        {64,64,256,16,16,32, 4,1},
#ifdef NO_OVERLAP
        {128,128,256,16,16,32, 4,1},
        {128,64,256,16,16,32, 4,1},
        {64,64,256,16,16,32, 4,1},
#endif
    };

extern std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *,workspace_t*)> >int4_conv_K64_split3x3_table;
extern std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *,workspace_t*)> >int4_conv_K128_split3x3_table;

extern std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *,workspace_t*)> >int4_conv_K256_split3x3_table;


std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K64_split3x3_table =
    {
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_split3x3_singlebuf_relu<128,128,64,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_split3x3_singlebuf_relu<256,64,64,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_split3x3_singlebuf_relu<128,64,64,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k64_split3x3_singlebuf_relu<64,64,64,16,16,16,4,1, true>,
    };

std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K128_split3x3_table =
    {
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_split3x3_singlebuf_relu<128,128,128,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_split3x3_singlebuf_relu<128,64,128,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k128_split3x3_singlebuf_relu<64,64,128,16,16,16,4,1, true>,
    };

std::vector<std::function<
    void(char*, char*, char*, int*,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        float*, bool, bool,
        uint8_t, float *, void *, workspace_t*)> > int4_conv_K256_split3x3_table =
    {
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_split3x3_singlebuf_relu<64,64,256,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_split3x3_singlebuf_relu<64,128,256,16,16,16,4,1, true>,
        lb_turing_indirect_conv_wmma_int4_ldg16_k256_split3x3_singlebuf_relu<128,128,256,16,16,16,4,1, true>,
    };


extern std::vector<std::vector<int> > int4_conv_K64_split3x3_table_tiling;
extern std::vector<std::vector<int> > int4_conv_K128_split3x3_table_tiling;
extern std::vector<std::vector<int> > int4_conv_K256_split3x3_table_tiling;


std::vector<std::vector<int>> int4_conv_K64_split3x3_table_tiling = {

    {128,128,64,16,16,16, 4,1},
    {256,64,64,16,16,16, 4,1},
    {128,64,64,16,16,16, 4,1},
    {64,64,64,16,16,16, 4,1},
};
std::vector<std::vector<int>> int4_conv_K128_split3x3_table_tiling = {

    {128,128,128,16,16,16, 4,1},
    {128,64,128,16,16,16, 4,1},
    {64,64,128,16,16,16, 4,1},

};

std::vector<std::vector<int>> int4_conv_K256_split3x3_table_tiling = {

    {64,64,256,16,16,16, 4,1},
    {64,128,256,16,16,16, 4,1},
    {128,128,256,16,16,16, 4,1},

};
#endif
