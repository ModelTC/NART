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

#ifndef IM2COL_HPP
#define IM2COL_HPP

#include <stdint.h>
#include <stdio.h>
#include <string.h>

template <typename T>
void col2im(
    const T *col, T *data, const size_t channel, const size_t height, const size_t width,
    const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
    const size_t stride_h, const size_t stride_w);
template <typename T>
void im2col(
    const T *data, T *col, const size_t channel, const size_t height, const size_t width,
    const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h, const uint16_t pad_w,
    const uint16_t stride_h, const uint16_t stride_w, const uint16_t hole_h, const uint16_t hole_w);

template <typename T>
void im2col_err(
    const T *data, T *col, const size_t channel, const size_t height, const size_t width,
    const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h, const uint16_t pad_w,
    const uint16_t stride_h, const uint16_t stride_w)
{
    size_t i, j, k, l, c;
    size_t number;
    size_t h_split_1, h_split_2, h_split_3, h_split_4;
    size_t w_split_1, w_split_2, w_split_3, w_split_4;
    size_t kernel_area, kernel_size, kernel_count;
    size_t input_area;
    size_t data_h_offset;
    size_t copy_count;
    T *col_temp_1, *col_temp_2, *col_temp_3;
    const T *data_temp_1, *data_temp_2, *data_temp_3, *data_temp_4;

    if ((int)pad_h - (int)kernel_h >= 0) {
        h_split_1 = (pad_h - kernel_h) / stride_h * stride_h + stride_h;
    } else {
        h_split_1 = 0;
    }
    if ((int)height + 2 * (int)pad_h - (int)kernel_h < 0) {
        h_split_2 = 0;
    } else {
        if ((int)pad_h < (int)height + 2 * (int)pad_h - (int)kernel_h) {
            h_split_2 = pad_h / stride_h * stride_h + stride_h;
        } else {
            h_split_2 = (height + 2 * pad_h - kernel_h) / stride_h * stride_h + stride_h;
        }
    }
    if ((int)height + (int)pad_h - (int)kernel_h >= 0) {
        h_split_3 = (height + pad_h - kernel_h) / stride_h * stride_h + stride_h;
    } else {
        h_split_3 = 0;
    }
    if ((int)height + 2 * (int)pad_h - (int)kernel_h < 0) {
        h_split_4 = 0;
    } else {
        if ((int)height + (int)pad_h < (int)height + 2 * (int)pad_h - (int)kernel_h) {
            h_split_4 = (height + pad_h) / stride_h * stride_h + stride_h;
        } else {
            h_split_4 = (height + 2 * pad_h - kernel_h) / stride_h * stride_h + stride_h;
        }
    }

    if ((int)pad_w - (int)kernel_w >= 0) {
        w_split_1 = (pad_w - kernel_w) / stride_w * stride_w + stride_w;
    } else {
        w_split_1 = 0;
    }
    if ((int)width + 2 * (int)pad_w - (int)kernel_w < 0) {
        w_split_2 = 0;
    } else {
        if ((int)pad_w < (int)width + 2 * (int)pad_w - (int)kernel_w) {
            w_split_2 = pad_w / stride_w * stride_w + stride_w;
        } else {
            w_split_2 = (width + 2 * pad_w - kernel_w) / stride_w * stride_w + stride_w;
        }
    }
    if ((int)width + (int)pad_w - (int)kernel_w >= 0) {
        w_split_3 = (width + pad_w - kernel_w) / stride_w * stride_w + stride_w;
    } else {
        w_split_3 = 0;
    }
    if ((int)width + 2 * (int)pad_w - (int)kernel_w < 0) {
        w_split_4 = 0;
    } else {
        if ((int)width + (int)pad_w < (int)width + 2 * (int)pad_w - (int)kernel_w) {
            w_split_4 = (width + pad_w) / stride_w * stride_w + stride_w;
        } else {
            w_split_4 = (width + 2 * pad_w - kernel_w) / stride_w * stride_w + stride_w;
        }
    }

    number = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    kernel_area = kernel_h * kernel_w;
    kernel_size = channel * kernel_area;
    kernel_count = number * kernel_size;
    input_area = height * width;
    data_h_offset = stride_h * width;
    copy_count = sizeof(T) * kernel_w;

    for (i = 0; i < h_split_1; i += stride_h) {
        // col_temp_1 = col;
        // for (j = 0; j < i + kernel_h; ++j) {
        //     col_temp_2 = col_temp_1;
        //     for (c = 0; c < channel; ++c) {
        //         col_temp_3 = col_temp_2;
        //         for (k = 0; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
        //             memset(col_temp_3, 0, sizeof(T) * kernel_w);
        //             col_temp_3 += kernel_size;
        //         }
        //         col_temp_2 += kernel_area;
        //     }
        //     col_temp_1 += kernel_w;
        // }
        col += kernel_count;
    }

    // h方向上跨越了第一个pad部分和data部分的kernel
    for (; i < h_split_2; i += stride_h) {
        col_temp_1 = col;
        data_temp_1 = data;
        for (j = i; j < pad_h; ++j) {
            // col_temp_2 = col_temp_1;
            // for (c = 0; c < channel; ++c) {
            //     col_temp_3 = col_temp_2;
            //     for (k = 0; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
            //         memset(col_temp_3, 0, sizeof(T) * kernel_w);
            //         col_temp_3 += kernel_size;
            //     }
            //     col_temp_2 += kernel_area;
            // }
            col_temp_1 += kernel_w;
        }
        for (; j < i + kernel_h; ++j) {
            col_temp_2 = col_temp_1;
            for (k = 0; k < w_split_1; k += stride_w) {
                // col_temp_3 = col_temp_2;
                // for (c = 0; c < channel; ++c) {
                //     memset(col_temp_3, 0, sizeof(T) * kernel_w);
                //     col_temp_3 += kernel_area;
                // }
                col_temp_2 += kernel_size;
            }
            // w方向上跨越第一个pad部分和data部分的kernel
            for (; k < w_split_2; k += stride_w) {
                data_temp_2 = data_temp_1;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_3 = data_temp_2;
                    for (l = k; l < pad_w; ++l) {
                        // 指针移动 赋值为0
                        *col_temp_3++ = 0;
                    }
                    for (; l < k + kernel_w; ++l) {
                        // 指针移动 取data中的数据
                        *col_temp_3++ = *data_temp_3++;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_2 += input_area;
                }
            }
            data_temp_2 = data_temp_1 + k - pad_w;
            for (; k < w_split_3; k += stride_w) {
                col_temp_3 = col_temp_2;
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    // memcpy(col_temp_3, data_temp_3, sizeof(T) * kernel_w);
                    memcpy(col_temp_3, data_temp_3, copy_count);
                    col_temp_3 += kernel_area;
                    data_temp_3 += input_area;
                }
                col_temp_2 += kernel_size;
                data_temp_2 += stride_w;
            }
            // w方向上跨越data部分和第二个pad部分的kernel
            for (; k < w_split_4; k += stride_w) {
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_4 = data_temp_3;
                    for (l = k; l < width + pad_w; ++l) {
                        // 指针移动 取data中的数据
                        *col_temp_3++ = *data_temp_4++;
                    }
                    for (; l < k + kernel_w; ++l) {
                        // 指针移动 赋值为0
                        *col_temp_3++ = 0;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_3 += height * width;
                }
                data_temp_2 += stride_w;
            }
            // for (; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
            //     col_temp_3 = col_temp_2;
            //     for (c = 0; c < channel; ++c) {
            //         memset(col_temp_3, 0, sizeof(T) * kernel_w);
            //         col_temp_3 += kernel_area;
            //     }
            //     col_temp_2 += kernel_size;
            // }
            col_temp_1 += kernel_w;
            data_temp_1 += width;
        }
        col += kernel_count;
    }
    data += (i - pad_h) * width;
    for (; i < h_split_3; i += stride_h) {
        col_temp_1 = col;
        data_temp_1 = data;
        for (j = i; j < i + kernel_h; ++j) {
            col_temp_2 = col_temp_1;
            for (k = 0; k < w_split_1; k += stride_w) {
                // col_temp_3 = col_temp_2;
                // for (c = 0; c < channel; ++c) {
                //     memset(col_temp_3, 0, sizeof(T) * kernel_w);
                //     col_temp_3 += kernel_area;
                // }
                col_temp_2 += kernel_size;
            }
            //跨越w方向上的一个pad部分和data部分的kernel
            for (; k < w_split_2; k += stride_w) {
                data_temp_2 = data_temp_1;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_3 = data_temp_2;
                    for (l = k; l < pad_w; ++l) {
                        // 指针移动 赋值为0
                        *col_temp_3++ = 0;
                    }
                    for (; l < k + kernel_w; ++l) {
                        // 指针移动 取data中的数据
                        *col_temp_3++ = *data_temp_3++;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_2 += input_area;
                }
            }
            data_temp_2 = data_temp_1 + k - pad_w;
            for (; k < w_split_3; k += stride_w) {
                col_temp_3 = col_temp_2;
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    // memcpy(col_temp_3, data_temp_3, sizeof(T) * kernel_w);
                    memcpy(col_temp_3, data_temp_3, copy_count);
                    col_temp_3 += kernel_area;
                    data_temp_3 += input_area;
                }
                col_temp_2 += kernel_size;
                data_temp_2 += stride_w;
            }
            // w方向上跨越data部分和第二个pad部分的kernel
            for (; k < w_split_4; k += stride_w) {
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_4 = data_temp_3;
                    for (l = k; l < width + pad_w; ++l) {
                        // 指针移动 取data中的数据
                        *col_temp_3++ = *data_temp_4++;
                    }
                    for (; l < k + kernel_w; ++l) {
                        // 指针移动 赋值为0
                        *col_temp_3++ = 0;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_3 += input_area;
                }
                data_temp_2 += stride_w;
            }
            // for (; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
            //     col_temp_3 = col_temp_2;
            //     for (c = 0; c < channel; ++c) {
            //         memset(col_temp_3, 0, sizeof(T) * kernel_w);
            //         col_temp_3 += kernel_area;
            //     }
            //     col_temp_2 += kernel_size;
            // }
            col_temp_1 += kernel_w;
            data_temp_1 += width;
        }
        col += kernel_count;
        data += data_h_offset;
    }
    // 跨越h方向上data部分和第二个pad部分的kernel
    for (; i < h_split_4; i += stride_h) {
        col_temp_1 = col;
        data_temp_1 = data;
        for (j = i; j < height + pad_w; ++j) {
            col_temp_2 = col_temp_1;
            for (k = 0; k < w_split_1; k += stride_w) {
                // col_temp_3 = col_temp_2;
                // for (c = 0; c < channel; ++c) {
                //     memset(col_temp_3, 0, sizeof(T) * kernel_w);
                //     col_temp_3 += kernel_area;
                // }
                col_temp_2 += kernel_size;
            }
            // 跨越w方向上第一个pad部分和data部分的kernel
            for (; k < w_split_2; k += stride_w) {
                data_temp_2 = data_temp_1;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_3 = data_temp_2;
                    for (l = k; l < pad_w; ++l) {
                        *col_temp_3++ = 0;
                    }
                    for (; l < k + kernel_w; ++l) {
                        *col_temp_3++ = *data_temp_3++;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_2 += input_area;
                }
            }
            data_temp_2 = data_temp_1 + k - pad_w;
            for (; k < w_split_3; k += stride_w) {
                col_temp_3 = col_temp_2;
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    memcpy(col_temp_3, data_temp_3, copy_count);
                    col_temp_3 += kernel_area;
                    data_temp_3 += input_area;
                }
                col_temp_2 += kernel_size;
                data_temp_2 += stride_w;
            }
            // w方向上跨越data部分和第二个pad部分的kernel
            for (; k < w_split_4; k += stride_w) {
                data_temp_3 = data_temp_2;
                for (c = 0; c < channel; ++c) {
                    col_temp_3 = col_temp_2;
                    data_temp_4 = data_temp_3;
                    for (l = k; l < width + pad_w; ++l) {
                        // 指针移动 取data中的数据
                        *col_temp_3++ = *data_temp_4++;
                    }
                    for (; l < k + kernel_w; ++l) {
                        // 指针移动 赋值为0
                        *col_temp_3++ = 0;
                    }
                    col_temp_2 += kernel_area;
                    data_temp_3 += input_area;
                }
                data_temp_2 += stride_w;
            }
            // for (; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
            //     col_temp_3 = col_temp_2;
            //     for (c = 0; c < channel; ++c) {
            //         memset(col_temp_3, 0, sizeof(T) * kernel_w);
            //         col_temp_3 += kernel_area;
            //     }
            //     col_temp_2 += kernel_size;
            // }
            col_temp_1 += kernel_w;
            data_temp_1 += width;
        }
        // for (; j < i + kernel_h; ++j) {
        //     col_temp_2 = col_temp_1;
        //     for (c = 0; c < channel; ++c) {
        //         col_temp_3 = col_temp_2;
        //         for (k = 0; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
        //             memset(col_temp_3, 0, sizeof(T) * kernel_w);
        //             col_temp_3 += kernel_size;
        //         }
        //         col_temp_2 += kernel_area;
        //     }
        //     col_temp_1 += kernel_w;
        // }
        col += kernel_count;
        data += data_h_offset;
    }
    // for (; i + kernel_h <= height + 2 * pad_h; i += stride_h) {
    //     col_temp_1 = col;
    //     for (j = 0; j < i + kernel_h; ++j) {
    //         col_temp_2 = col_temp_1;
    //         for (c = 0; c < channel; ++c) {
    //             col_temp_3 = col_temp_2;
    //             for (k = 0; k + kernel_w <= width + 2 * pad_w; k += stride_w) {
    //                 memset(col_temp_3, 0, sizeof(T) * kernel_w);
    //                 col_temp_3 += kernel_size;
    //             }
    //             col_temp_2 += kernel_area;
    //         }
    //         col_temp_1 += kernel_w;
    //     }
    //     col += kernel_count;
    // }
}

template <typename T>
void im2col(
    const T *data, T *col, const size_t channel, const size_t height, const size_t width,
    const uint16_t kernel_h, const uint16_t kernel_w, const uint16_t pad_h, const uint16_t pad_w,
    const uint16_t stride_h, const uint16_t stride_w, const uint16_t hole_h, const uint16_t hole_w)
{
    T *temp = col;
    size_t h, w, k, c;
    size_t kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
    size_t kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
    size_t height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
    size_t width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
    for (h = 0; h < height_col; ++h) {
        int h_padt = h * stride_h - pad_h;
        for (w = 0; w < width_col; ++w) {
            int w_padt = w * stride_w - pad_w;
            for (c = 0; c < channel; ++c) {
                for (k = 0; k < (unsigned int)kernel_h * kernel_w; k++) {
                    int h_pad = h_padt + k / kernel_w % kernel_h * hole_h;
                    int w_pad = w_padt + k % kernel_w * hole_w;
                    int a = h_pad >= 0 && h_pad < (int)height && w_pad >= 0 && w_pad < (int)width;
                    if (a)
                        *temp = data[(c * height + h_pad) * width + w_pad];
                    else
                        *temp = 0;
                    temp++;
                }
            }
        }
    }
    return;
}

template <typename T>
void col2im(
    const T *col, T *data, const size_t channel, const size_t height, const size_t width,
    const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
    const size_t stride_h, const size_t stride_w)
{
    size_t c, h, w;
    size_t col_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t col_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    size_t col_channel = channel * kernel_h * kernel_w;

    for (c = 0; c < col_channel; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_data = c / kernel_h / kernel_w;
        for (h = 0; h < col_height; ++h) {
            for (w = 0; w < col_width; ++w) {
                int h_pad = (int)h * (int)stride_h - (int)pad_h + h_offset;
                int w_pad = (int)w * (int)stride_w - (int)pad_w + w_offset;
                if (h_pad >= 0 && h_pad < (int)height && w_pad >= 0 && w_pad < (int)width) {
                    data[(c_data * (int)height + h_pad) * (int)width + w_pad]
                        += col[(c * (int)col_height + h) * (int)col_width + w];
                }
            }
        }
    }
}
#endif
