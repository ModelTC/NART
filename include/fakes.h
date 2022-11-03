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

#include <memory>

#include "fake.h"

namespace nart {
using namespace std;
function<vector<pair<fake_op::dtype, fake_tensor::shape_t>>(
    const fake_op *, vector<fake_tensor::shape_t>)>
wrap_op_infer(const op_tp_t *op_tp, uint32_t input_tp);

class fake_conv_2d : public fake_op {
public:
    fake_conv_2d(
        bool bias, bool relu, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int channel_in, int channel_out, int group,
        const vector<std::shared_ptr<fake_tensor>> &input_tensors,
        const vector<std::shared_ptr<fake_tensor>> &param_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    bool bias_;
    bool relu_;
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    int channel_in_;
    int channel_out_;
    int group_;
};

class fake_pool : public fake_op {
public:
    enum pool_method {
        MAX = 0,
        AVE = 1,
    };
    fake_pool(
        pool_method method, bool ceil_mode, int kernel_h, int kernel_w, int stride_h, int stride_w,
        int pad_h, int pad_w, const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    pool_method method_;
    bool ceil_mode_;
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
};

class fake_relu : public fake_op {
public:
    fake_relu(const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_ip : public fake_op {
public:
    fake_ip(
        bool bias, bool relu, int channel_out, const vector<shared_ptr<fake_tensor>> &input_tensors,
        const vector<shared_ptr<fake_tensor>> &param_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    int channel_out_;
    int bias_;
    int relu_;
};

class fake_lrn : public fake_op {
public:
    fake_lrn(
        uint32_t local_size, float alpha, float beta, float k, uint32_t norm_region,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    uint32_t local_size_;
    float alpha_;
    float beta_;
    float k_;
    uint32_t norm_region_;
};

class fake_eltwise : public fake_op {
public:
    fake_eltwise(
        uint32_t opr, std::vector<float> coeff,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_bn : public fake_op {
public:
    fake_bn(
        float eps, const vector<shared_ptr<fake_tensor>> &input_tensors,
        const vector<shared_ptr<fake_tensor>> &param_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_deconv_2d : public fake_op {
public:
    fake_deconv_2d(
        bool bias, bool relu, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
        int pad_w, int channel_in, int channel_out, int group,
        const vector<std::shared_ptr<fake_tensor>> &input_tensors,
        const vector<std::shared_ptr<fake_tensor>> &param_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    bool bias_;
    bool relu_;
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    int channel_in_;
    int channel_out_;
    int group_;
};

class fake_softmax : public fake_op {
public:
    fake_softmax(uint32_t axis, const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_concat : public fake_op {
public:
    fake_concat(uint32_t axis, const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_sigmoid : public fake_op {
public:
    fake_sigmoid(const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_reshape : public fake_op {
public:
    fake_reshape(
        int32_t axis, int32_t num_axes, std::vector<int32_t> dims,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_interp : public fake_op {
public:
    fake_interp(
        uint32_t height, uint32_t width, uint32_t zoom_factor, uint32_t shrink_factor,
        uint32_t pad_beg, uint32_t pad_end, const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_prelu : public fake_op {
public:
    fake_prelu(
        bool share, const vector<shared_ptr<fake_tensor>> &input_tensors,
        const vector<shared_ptr<fake_tensor>> &param_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_quantize : public fake_op {
public:
    fake_quantize(
        float scale, uint8_t zero_point, uint8_t bits, int qtype,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_dequantize : public fake_op {
public:
    fake_dequantize(
        float scale, uint8_t zero_point, uint8_t bits, int qtype,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
};

class fake_pad : public fake_op {
public:
    fake_pad(
        const std::string &mode, const float value, const std::vector<int32_t> pads,
        const vector<shared_ptr<fake_tensor>> &input_tensors);
    virtual uint64_t _get_op_tp_code() const override;
    std::string mode_;
    float value_;
    std::vector<int32_t> pads_;
};

}
