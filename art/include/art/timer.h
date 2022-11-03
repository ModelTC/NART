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

#ifndef _TIMER_H_
#define _TIMER_H_

#ifdef TIO
#ifdef __linux__

#include <unistd.h>

#include "sys/time.h"

#define TI(tag)                                      \
    struct timeval time_start_##tag, time_end_##tag; \
    do {                                             \
        gettimeofday(&time_start_##tag, NULL);       \
    } while (0)

#define TO(tag, desc)                                                         \
    do {                                                                      \
        gettimeofday(&time_end_##tag, NULL);                                  \
        float msec = (time_end_##tag.tv_sec - time_start_##tag.tv_sec) * 1000 \
            + (time_end_##tag.tv_usec - time_start_##tag.tv_usec) / 1000.;    \
        printf("%s: %.4fms\n", desc, msec);                                   \
    } while (0)
#else
#define TI(tag) double __timing_start##tag = clock()

#define TO(tag, desc)                                                           \
    do {                                                                        \
        double __timing_end##tag = clock();                                     \
        fprintf(                                                                \
            stdout, "%s %lfms\n", desc,                                         \
            (__timing_end##tag - __timing_start##tag) / CLOCKS_PER_SEC * 1000); \
    } while (0)
#endif
#else
#define TI(tag)
#define TO(tag, desc)
#endif

#endif //_TIMER_H_
