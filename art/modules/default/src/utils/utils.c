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

#include "./utils.h"

float bilinear_interpolate(
    const float *input, const size_t height, const size_t width, float y, float x)
{
    if (y < -0.5 || y > height - 0.5 || x < -0.5 || x > width - 0.5) {
        // empty
        return 0.0f;
    }

    if (y <= 0) {
        y = 0.f;
    }

    if (x <= 0) {
        x = 0.f;
    }

    size_t y_low = (size_t)y;
    size_t x_low = (size_t)x;
    size_t y_high;
    size_t x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.f - ly, hx = 1.f - lx;
    float v1 = input[y_low * width + x_low];
    float v2 = input[y_low * width + x_high];
    float v3 = input[y_high * width + x_low];
    float v4 = input[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}
