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

#include <iostream>
#include <signal.h>
#include <sstream>

#include "art/op_settings.h"
#include "art/op_tp.h"

#include "include/fakes.h"

using namespace std;
using namespace nart;

uint64_t fake_bn::_get_op_tp_code() const { return OP_BN; }
uint64_t fake_conv_2d::_get_op_tp_code() const { return OP_CONV_2D; }
uint64_t fake_deconv_2d::_get_op_tp_code() const { return OP_DECONV_2D; }
uint64_t fake_eltwise::_get_op_tp_code() const { return OP_ELTWISE; }
uint64_t fake_ip::_get_op_tp_code() const { return OP_IP; }
uint64_t fake_lrn::_get_op_tp_code() const { return OP_LRN; }
uint64_t fake_pool::_get_op_tp_code() const { return OP_POOL; }
uint64_t fake_relu::_get_op_tp_code() const { return OP_RELU; }
uint64_t fake_softmax::_get_op_tp_code() const { return OP_SOFTMAX; }
uint64_t fake_concat::_get_op_tp_code() const { return OP_CONCAT; }
uint64_t fake_sigmoid::_get_op_tp_code() const { return OP_SIGMOID; }
uint64_t fake_reshape::_get_op_tp_code() const { return OP_RESHAPE; }
uint64_t fake_interp::_get_op_tp_code() const { return OP_INTERP; }
uint64_t fake_prelu::_get_op_tp_code() const { return OP_PRELU; }
uint64_t fake_pad::_get_op_tp_code() const { return OP_PAD; }

extern "C" const mem_tp *cpu_mem_tp;
function<vector<pair<fake_op::dtype, fake_tensor::shape_t>>(
    const fake_op *, vector<fake_tensor::shape_t>)>
nart::wrap_op_infer(const op_tp_t *op_tp, uint32_t input_tp)
{
    return [=](const fake_op *fop, vector<fake_tensor::shape_t> in) {
        vector<pair<fake_op::dtype, fake_tensor::shape_t>> res;
        op_t *op = (op_t *)malloc(sizeof(op_t));
        memset(op, 0, sizeof(op_t));
        op->input_size = in.size();
        op->input_tensors = (tensor_t **)malloc(sizeof(tensor_t *) * op->input_size);
        for (size_t i = 0; i < in.size(); ++i) {
            op->input_tensors[i] = tensor_new(cpu_mem_tp, 0);
            op->input_tensors[i]->dtype = input_tp;
            op->input_tensors[i]->shape.dim_size = in[i].dims.size();
            op->input_tensors[i]->shape.channel_axis = in[i].channel_axis;
            op->input_tensors[i]->shape.batch_axis = in[i].batch_axis;
            for (size_t j = 0; j < in[i].dims.size(); ++j) {
                op->input_tensors[i]->shape.dim[j] = in[i].dims[j];
            }
        }

        op_tp_entry_t entry;
        entry.tp = op_tp;
        op->entry = &entry;
        op->setting = setting_new();

        auto r = [](const fake_op::fake_setting *setting) {
            union {
                uint8_t u8;
                uint16_t u16;
                uint32_t u32;
                uint64_t u64;
                float f32;
                double f64;
                bool b;
            } res;
            ostringstream ss;
            setting->serialize(ss);
            memcpy(&res, ss.str().data(), min(sizeof(res), ss.str().length()));
            return res;
        };
        for (const auto &item : fop->get_settings()) {
            if (false == item.second->is_repeated()) {
                auto dtype = item.second->get_dtype();
                switch (dtype) {
                case dtUINT8:
                case dtINT8:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).u8));
                    break;
                case dtUINT16:
                case dtINT16:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).u16));
                    break;
                case dtUINT32:
                case dtINT32:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).u32));
                    break;
                case dtBOOL:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).b));
                    break;
                case dtFLOAT16:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).u16));
                    break;
                case dtFLOAT32:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).f32));
                    break;
                case dtFLOAT64:
                    CHECK(op_setting_single_set(op, item.first, dtype, r(item.second.get()).f64));
                    break;
                case dtSTR: {
                    CHECK(op_setting_single_set(
                        op, item.first, dtype,
                        ((fake_op::fake_setting_tp<string> *)item.second.get())->value_.data()));
                    break;
                }
                default:
                    CHECK(false);
                }
            } else {
                auto dtype = item.second->get_dtype();
                ostringstream ss;
                item.second->serialize(ss);
                const string &str = ss.str();
                size_t len = (str.length() - sizeof(uint32_t)) / datatype_sizeof(dtype);
                if (len == 0) {
                    LOG_error("The setting #%d of op `%s` is empty\n", item.first, op_tp->name);
                }
                CHECK(op_setting_array_set(
                    op, item.first, dtype, len, str.data() + sizeof(uint32_t)));
            }
        }
        op_tp->infer_output_func(op);

        for (size_t i = 0; i < op->output_size; ++i) {
            fake_tensor::shape_t sp;
            sp.dims.clear();
            for (int j = 0; j < op->output_tensors[i].shape.dim_size; ++j) {
                sp.dims.push_back(op->output_tensors[i].shape.dim[j]);
            }
            sp.channel_axis = op->output_tensors[i].shape.channel_axis;
            sp.batch_axis = op->output_tensors[i].shape.batch_axis;
            res.push_back(make_pair(op->output_tensors[i].dtype, sp));
        }

        for (size_t i = 0; i < op->input_size; ++i) {
            tensor_delete(op->input_tensors[i]);
        }
        op_destroy_default(op);
        free(op);
        return res;
    };
}

extern "C" op_tp_t op_pad;
fake_pad::fake_pad(
    const std::string &mode, const float value, const std::vector<int32_t> pads,
    const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {}), mode_(mode), value_(value), pads_(pads)
{
    set_setting(SETTING_PAD_MODE, unique_ptr<fake_setting>(new fake_setting_tp<std::string>(mode)));
    set_setting(SETTING_PAD_VALUE, unique_ptr<fake_setting>(new fake_setting_tp<float>(value)));
    set_setting(
        SETTING_PAD_PADS, unique_ptr<fake_setting>(new fake_setting_repeat_tp<int32_t>(pads)));
}

extern "C" op_tp_t op_conv_2d_tp;
fake_conv_2d::fake_conv_2d(
    bool bias, bool relu, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int channel_in, int channel_out, int group,
    const vector<std::shared_ptr<fake_tensor>> &input_tensors,
    const vector<std::shared_ptr<fake_tensor>> &param_tensors)
    : fake_op(input_tensors, param_tensors)
    , bias_(bias)
    , relu_(relu)
    , kernel_h_(kernel_h)
    , kernel_w_(kernel_w)
    , stride_h_(stride_h)
    , stride_w_(stride_w)
    , pad_h_(pad_h)
    , pad_w_(pad_w)
    , channel_in_(channel_in)
    , channel_out_(channel_out)
    , group_(group)

{
    set_setting(
        SETTING_CONV_2D_NUM_OUTPUT,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(channel_out)));
    set_setting(
        SETTING_CONV_2D_PAD_H, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_h)));
    set_setting(
        SETTING_CONV_2D_PAD_W, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_w)));
    set_setting(
        SETTING_CONV_2D_KERNEL_H,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_h)));
    set_setting(
        SETTING_CONV_2D_KERNEL_W,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_w)));
    set_setting(
        SETTING_CONV_2D_STRIDE_H,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_h)));
    set_setting(
        SETTING_CONV_2D_STRIDE_W,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_w)));
    set_setting(
        SETTING_CONV_2D_GROUP, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(group)));
    set_setting(
        SETTING_CONV_2D_RELU_FLAG, unique_ptr<fake_setting>(new fake_setting_tp<bool>(relu)));
    inferer_ = wrap_op_infer(&op_conv_2d_tp, input_tensors[0]->dtype());
}

extern "C" op_tp_t op_pool_tp;
fake_pool::fake_pool(
    fake_pool::pool_method method, bool ceil_mode, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int pad_h, int pad_w, const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
    , method_(method)
    , ceil_mode_(ceil_mode)
    , kernel_h_(kernel_h)
    , kernel_w_(kernel_w)
    , stride_h_(stride_h)
    , stride_w_(stride_w)
    , pad_h_(pad_h)
    , pad_w_(pad_w)
{
    set_setting(SETTING_POOL_PAD_H, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_h)));
    set_setting(SETTING_POOL_PAD_W, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_w)));
    set_setting(
        SETTING_POOL_KERNEL_H, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_h)));
    set_setting(
        SETTING_POOL_KERNEL_W, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_w)));
    set_setting(
        SETTING_POOL_STRIDE_H, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_h)));
    set_setting(
        SETTING_POOL_STRIDE_W, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_w)));
    set_setting(
        SETTING_POOL_CEIL_MODE, unique_ptr<fake_setting>(new fake_setting_tp<bool>(ceil_mode)));
    set_setting(
        SETTING_POOL_METHOD, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(method)));
    inferer_ = wrap_op_infer(&op_pool_tp, input_tensors[0]->dtype());
}

fake_relu::fake_relu(const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
}

extern "C" op_tp_t op_ip_tp;
fake_ip::fake_ip(
    bool bias, bool relu, int channel_out, const vector<shared_ptr<fake_tensor>> &input_tensors,
    const vector<std::shared_ptr<fake_tensor>> &param_tensors)
    : fake_op(input_tensors, param_tensors), channel_out_(channel_out), bias_(bias), relu_(relu)
{
    set_setting(
        SETTING_IP_NUM_OUTPUT,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(channel_out)));
    inferer_ = wrap_op_infer(&op_ip_tp, input_tensors[0]->dtype());
}

fake_lrn::fake_lrn(
    uint32_t local_size, float alpha, float beta, float k, uint32_t norm_region,
    const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
    , local_size_(local_size)
    , alpha_(alpha)
    , beta_(beta)
    , k_(k)
    , norm_region_(norm_region)
{
    set_setting(
        SETTING_LRN_LOCAL_SIZE,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(local_size)));
    set_setting(SETTING_LRN_ALPHA, unique_ptr<fake_setting>(new fake_setting_tp<float>(alpha)));
    set_setting(SETTING_LRN_BETA, unique_ptr<fake_setting>(new fake_setting_tp<float>(beta)));
    set_setting(SETTING_LRN_K, unique_ptr<fake_setting>(new fake_setting_tp<float>(k)));
    set_setting(
        SETTING_LRN_NORM_REGION,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(norm_region)));
}

fake_eltwise::fake_eltwise(
    uint32_t opr, std::vector<float> coeff, const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
    set_setting(
        SETTING_ELTWISE_OPERATION, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(opr)));
    if (coeff.size() > 0)
        set_setting(
            SETTING_ELTWISE_COEFF,
            unique_ptr<fake_setting>(new fake_setting_repeat_tp<float>(coeff)));
}

fake_bn::fake_bn(
    float eps, const vector<shared_ptr<fake_tensor>> &input_tensors,
    const vector<shared_ptr<fake_tensor>> &param_tensors)
    : fake_op(input_tensors, param_tensors)
{
    set_setting(SETTING_BN_EPS, unique_ptr<fake_setting>(new fake_setting_tp<float>(eps)));
}

extern "C" op_tp_t op_deconv_2d_tp;
fake_deconv_2d::fake_deconv_2d(
    bool bias, bool relu, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int channel_in, int channel_out, int group,
    const vector<std::shared_ptr<fake_tensor>> &input_tensors,
    const vector<std::shared_ptr<fake_tensor>> &param_tensors)
    : fake_op(input_tensors, param_tensors)
    , bias_(bias)
    , relu_(relu)
    , kernel_h_(kernel_h)
    , kernel_w_(kernel_w)
    , stride_h_(stride_h)
    , stride_w_(stride_w)
    , pad_h_(pad_h)
    , pad_w_(pad_w)
    , channel_in_(channel_in)
    , channel_out_(channel_out)
    , group_(group)

{
    set_setting(
        SETTING_CONV_2D_NUM_OUTPUT,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(channel_out)));
    set_setting(
        SETTING_CONV_2D_PAD_H, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_h)));
    set_setting(
        SETTING_CONV_2D_PAD_W, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_w)));
    set_setting(
        SETTING_CONV_2D_KERNEL_H,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_h)));
    set_setting(
        SETTING_CONV_2D_KERNEL_W,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(kernel_w)));
    set_setting(
        SETTING_CONV_2D_STRIDE_H,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_h)));
    set_setting(
        SETTING_CONV_2D_STRIDE_W,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(stride_w)));
    set_setting(
        SETTING_CONV_2D_GROUP, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(group)));
    inferer_ = wrap_op_infer(&op_deconv_2d_tp, input_tensors[0]->dtype());
}

fake_softmax::fake_softmax(uint32_t axis, const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
    set_setting(
        SETTING_SOFTMAX_AXIS, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(axis)));
}

extern "C" op_tp_t op_concat_tp;
fake_concat::fake_concat(uint32_t axis, const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
    set_setting(SETTING_CONCAT_AXIS, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(axis)));
    inferer_ = wrap_op_infer(&op_concat_tp, input_tensors[0]->dtype());
}

fake_sigmoid::fake_sigmoid(const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
}

extern "C" op_tp_t op_reshape_tp;
fake_reshape::fake_reshape(
    int32_t axis, int32_t num_axes, std::vector<int32_t> dims,
    const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
    set_setting(SETTING_RESHAPE_AXIS, unique_ptr<fake_setting>(new fake_setting_tp<int32_t>(axis)));
    set_setting(
        SETTING_RESHAPE_NUM_AXES, unique_ptr<fake_setting>(new fake_setting_tp<int32_t>(num_axes)));
    set_setting(
        SETTING_RESHAPE_DIMS, unique_ptr<fake_setting>(new fake_setting_repeat_tp<int32_t>(dims)));
    inferer_ = wrap_op_infer(&op_reshape_tp, input_tensors[0]->dtype());
}

extern "C" op_tp_t op_interp_tp;
fake_interp::fake_interp(
    uint32_t height, uint32_t width, uint32_t zoom_factor, uint32_t shrink_factor, uint32_t pad_beg,
    uint32_t pad_end, const vector<shared_ptr<fake_tensor>> &input_tensors)
    : fake_op(input_tensors, {})
{
    set_setting(
        SETTING_INTERP_HEIGHT, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(height)));
    set_setting(
        SETTING_INTERP_WIDTH, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(width)));
    set_setting(
        SETTING_INTERP_ZOOM_FACTOR,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(zoom_factor)));
    set_setting(
        SETTING_INTERP_SHRINK_FACTOR,
        unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(shrink_factor)));
    set_setting(
        SETTING_INTERP_PAD_BEG, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_beg)));
    set_setting(
        SETTING_INTERP_PAD_END, unique_ptr<fake_setting>(new fake_setting_tp<uint32_t>(pad_end)));
    inferer_ = wrap_op_infer(&op_interp_tp, input_tensors[0]->dtype());
}

fake_prelu::fake_prelu(
    bool share, const vector<shared_ptr<fake_tensor>> &input_tensors,
    const vector<shared_ptr<fake_tensor>> &param_tensors)
    : fake_op(input_tensors, param_tensors)
{
    set_setting(SETTING_PRELU_SHARE, unique_ptr<fake_setting>(new fake_setting_tp<bool>(share)));
}
