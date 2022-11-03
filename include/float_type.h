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

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float float32_t;
#if defined(__aarch64__) && __GNUC__ >= 5
typedef __fp16 float16_t;
static inline float32_t fp16_to_fp32(float16_t v) { return (float32_t)v; }
static inline float16_t fp32_to_fp16(float32_t v) { return (float16_t)v; }
#else
struct float16_t;
static inline float32_t fp16_to_fp32(float16_t v);
static inline float16_t fp32_to_fp16(float32_t v);
typedef struct float16_t {
    uint16_t val;
    float16_t() { val = 0; }
    float16_t(const float32_t &v) { *this = fp32_to_fp16(v); }
    float16_t(const uint32_t &v) { val = v; }
    float16_t(const uint16_t &v) { val = v; }

    operator float32_t() const { return fp16_to_fp32(*this); }
} float16_t;
/*
 * [x] these two convert function refer to https://gist.github.com/martinkallman/5049614
 * [o] these two convert function refer to
 * numpy(https://github.com/numpy/numpy/blob/master/numpy/core/src/npymath/halffloat.c)
 */
static inline float32_t fp16_to_fp32(float16_t h)
{
    uint16_t h_exp, h_sig;
    uint32_t f_sgn, f_exp, f_sig;
    union {
        float32_t f32;
        uint32_t u32;
    } res;

    h_exp = (h.val & 0x7c00u);
    f_sgn = ((uint32_t)h.val & 0x8000u) << 16;
    switch (h_exp) {
    case 0x0000u: /* 0 or subnormal */
        h_sig = (h.val & 0x03ffu);
        /* Signed zero */
        if (h_sig == 0) {
            return f_sgn;
        }
        /* Subnormal */
        h_sig <<= 1;
        while ((h_sig & 0x0400u) == 0) {
            h_sig <<= 1;
            h_exp++;
        }
        f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
        f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
        res.u32 = f_sgn + f_exp + f_sig;
    case 0x7c00u: /* inf or NaN */
        /* All-ones exponent and a copy of the significand */
        res.u32 = f_sgn + 0x7f800000u + (((uint32_t)(h.val & 0x03ffu)) << 13);
    default: /* normalized */
        /* Just need to adjust the exponent and shift */
        res.u32 = f_sgn + (((uint32_t)(h.val & 0x7fffu) + 0x1c000u) << 13);
    }
    return res.f32;
}

static inline float16_t fp32_to_fp16(float32_t v)
{
    uint32_t f_exp, f_sig;
    uint32_t f = *(uint32_t *)&v;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (uint16_t)((f & 0x80000000u) >> 16);
    f_exp = (f & 0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f & 0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return (uint16_t)(h_sgn + ret);
            } else {
                /* signed inf */
                return (uint16_t)(h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
            //#if NPY_HALF_GENERATE_OVERFLOW
            // npy_set_floatstatus_overflow();
            //#endif
            return (uint16_t)(h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
            //#if NPY_HALF_GENERATE_UNDERFLOW
            /* If f != 0, it underflowed to 0 */
            // if ((f&0x7fffffff) != 0) {
            //     npy_set_floatstatus_underflow();
            // }
            //#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f & 0x007fffffu));
        //#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        // if ((f_sig&(((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
        //     npy_set_floatstatus_underflow();
        // }
        //#endif
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        //#if NPY_HALF_ROUND_TIES_TO_EVEN
        //        /*
        //         * If the last bit in the half significand is 0 (already even), and
        //         * the remaining bit pattern is 1000...0, then we do not add one
        //         * to the bit after the half significand.  In all other cases, we do.
        //         */
        //        if ((f_sig&0x00003fffu) != 0x00001000u) {
        //            f_sig += 0x00001000u;
        //        }
        //#else
        f_sig += 0x00001000u;
        //#endif
        h_sig = (uint16_t)(f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (uint16_t)(h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f & 0x007fffffu);
    //#if NPY_HALF_ROUND_TIES_TO_EVEN
    //    /*
    //     * If the last bit in the half significand is 0 (already even), and
    //     * the remaining bit pattern is 1000...0, then we do not add one
    //     * to the bit after the half significand.  In all other cases, we do.
    //     */
    //    if ((f_sig&0x00003fffu) != 0x00001000u) {
    //        f_sig += 0x00001000u;
    //    }
    //#else
    f_sig += 0x00001000u;
    //#endif
    h_sig = (uint16_t)(f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    //#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    // if (h_sig == 0x7c00u) {
    //     npy_set_floatstatus_overflow();
    // }
    return (uint16_t)(h_sgn + h_sig);
    //#else
    //    return h_sgn + h_exp + h_sig;
    //#endif
}

#endif // aarch64

#ifdef __cplusplus
}
#endif
