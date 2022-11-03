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

#include "art/compat.h"

#if defined(_MSC_VER)

#include <psapi.h>
#include <windows.h>

void nart_clocktime(struct timespec *ts)
{
    FILETIME ft;
    ULARGE_INTEGER s;
    ULONGLONG t;

    GetSystemTimeAsFileTime(&ft);
    s.LowPart = ft.dwLowDateTime;
    s.HighPart = ft.dwHighDateTime;
    t = s.QuadPart - 116444736000000000ULL;
    ts->tv_sec = t / 10000000;
    ts->tv_nsec = (t % 10000000) * 100;
}

#else

void nart_clocktime(struct timespec *ts) { clock_gettime(CLOCK_REALTIME, ts); }
#endif

#ifdef __UCLIBC__
float fmaxf(float a, float b) { return a > b ? a : b; }
float fminf(float a, float b) { return a < b ? a : b; }
#endif
