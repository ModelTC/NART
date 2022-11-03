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

#ifndef QUANT_HELPER_H
#define QUANT_HELPER_H

#include <math.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define USE_FIXED_POINT_ONLY

static inline int clip(int x, int l, int h) { return x > h ? h : (x < l ? l : x); }

static inline uint8_t saturate_int_by_bits(int32_t x, uint8_t bits)
{
    return x > ((1 << bits) - 1) ? ((1 << bits) - 1) : (x < 0 ? 0 : (uint8_t)(x));
}

static inline uint8_t saturate_float_by_bits(float x, uint8_t bits)
{
    return x > ((1 << bits) - 1) ? ((1 << bits) - 1) : (x < 0 ? 0 : (uint8_t)(x + 0.5));
}

static inline int8_t ssaturate_int_by_bits(int32_t x, uint8_t bits)
{
    int mval = (1 << bits) - 1;
    return clip(x, -mval, mval);
}

static inline int8_t ssaturate_float_by_bits(float x, uint8_t bits)
{
    int mval = (1 << bits) - 1;
    return clip(round(x), -mval, mval);
}

static inline int64_t rshift_rn(int64_t x, uint8_t bits)
{
    return ((x >> (bits - 1)) + (((x >> (bits - 1)) & 1) << 1)) >> 1;
}

static inline void get_quant_scale_int32(float x, uint32_t *multer, int8_t *shift)
{
    *shift = 0;
    if (fabs(x - 0) < 1e-10) {
        *multer = 0;
        return;
    }
    while (x >= 1.0f) {
        x /= 2;
        *shift -= 1;
    }
    while (x < 0.5f) {
        x *= 2;
        *shift += 1;
    }
    *shift += 31;
    *multer = x * (1ll << 31);
    return;
}

static inline void get_quant_scale_int16(float x, uint16_t *multer, int8_t *shift)
{
    *shift = 0;
    if (fabs(x - 0) < 1e-10) {
        *multer = 0;
        return;
    }
    while (x >= 1.0f) {
        x /= 2;
        *shift -= 1;
    }
    while (x < 0.5f) {
        x *= 2;
        *shift += 1;
    }
    *shift += 15;
    *multer = x * (1l << 15);
    return;
}

static inline void get_quant_scale(float x, uint32_t *multer, int8_t *shift, int bits)
{
    *shift = 0;
    if (fabs(x - 0) < 1e-10) {
        *multer = 0;
        return;
    }
    while (x >= 1.0f) {
        x /= 2;
        *shift -= 1;
    }
    while (x < 0.5f) {
        x *= 2;
        *shift += 1;
    }
    *shift += bits - 1;
    *multer = x * (1l << (bits - 1));
    return;
}
#endif // QUANT_HELPER_H
