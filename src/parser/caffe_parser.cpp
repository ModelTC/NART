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

#include "art/data_type.h"
#include "art/log.h"

#include "include/fakes.h"
#include "include/parser/caffe_parser.h"

using namespace nart;
using namespace std;
using namespace caffe;

map<string, vector<int>> nart::get_input(const NetParameter &netdef)
{
    int input_size = netdef.input_size();
    map<string, vector<int>> res;
    CHECK(input_size * 4 == netdef.input_dim_size() || input_size == netdef.input_shape_size());
    if (input_size * 4 == netdef.input_dim_size()) {
        for (int i = 0; i < input_size; ++i) {
            vector<int> v;
            for (int j = 0; j < 4; ++j) {
                v.push_back(netdef.input_dim(4 * i + j));
            }
            res[netdef.input(i)] = v;
        }
    } else {
        for (int i = 0; i < input_size; ++i) {
            vector<int> v;
            const auto &shape = netdef.input_shape(i);
            for (int j = 0; j < shape.dim_size(); ++j) {
                v.push_back(shape.dim(j));
            }
            res[netdef.input(i)] = v;
        }
    }
    return res;
}

const LayerParameter *nart::find_weight(const string &name, const NetParameter &weight)
{
    for (int i = 0; i < weight.layer_size(); ++i) {
        if (name == weight.layer(i).name())
            return &weight.layer(i);
    }
    return nullptr;
}

unique_ptr<fake_parade> caffe_parser::parse()
{
    unique_ptr<fake_parade> parade(new fake_parade);
    /* input */
    for (const auto &item : get_input(netdef_)) {
        if (mp_tensor_.end() == mp_tensor_.find(item.first)) {
            auto tensor = init_input_func_(item.first, item.second);
            mp_tensor_[item.first] = tensor;
        }
    }
    for (int i = 0; i < netdef_.layer_size(); ++i) {
        const LayerParameter &layer = netdef_.layer(i);
        const LayerParameter *weight = find_weight(layer.name(), netweight_);
        // CHECK_NE(nullptr, weight);
        if (mp_parse_func_.end() != mp_parse_func_.find(layer.type())) {
            auto op = mp_parse_func_[layer.type()](layer, *weight);
            auto pop = parade->append(std::move(op));
            auto outputs = pop->get_output_tensors();
            CHECK_EQ((int)layer.top_size(), (int)outputs.size());
            for (int i = 0; i < layer.top_size(); ++i) {
                outputs[i]->name = layer.top(i);
                mp_tensor_[outputs[i]->name] = outputs[i];
            }
        } else {
            fprintf(stderr, "Unsupported Layer: %s\n", layer.type().c_str());
            CHECK(false);
        }
    }
    return parade;
}

caffe_parser::caffe_parser(const NetParameter &netdef, const NetParameter &netweight)
    : netdef_(netdef), netweight_(netweight)
{
    auto gather_layer_input = [=](const LayerParameter &layerdef) {
        vector<shared_ptr<fake_tensor>> res;
        for (int i = 0; i < layerdef.bottom_size(); ++i) {
            CHECK_NE(mp_tensor_.end(), mp_tensor_.find(layerdef.bottom(i)));
            res.push_back(mp_tensor_[layerdef.bottom(i)]);
        }
        return res;
    };
    auto gather_layer_param
        = [=](const LayerParameter &layerweight) { return init_weight_func_(layerweight); };
    /* conv_2d */
    mp_parse_func_["Convolution"] = [=](const LayerParameter &layerdef,
                                        const LayerParameter &layerweight) {
        const ConvolutionParameter &conv = layerdef.convolution_param();
        auto inputs = gather_layer_input(layerdef);
        auto weights = gather_layer_param(layerweight);
        int dim_h = 2;
        int dim_w = 3;
        int dim_c = 1;
        int dim_n = 0;
        if (weights[0]->shape.batch_axis == 3) {
            dim_h = 0;
            dim_w = 1;
            dim_c = 2;
            dim_n = 3;
        }
        unique_ptr<fake_op> res(new fake_conv_2d(
            conv.bias_term(), false, weights[0]->shape.dims[dim_h], weights[0]->shape.dims[dim_w],

            conv.has_stride_h() ? conv.stride_h() : conv.stride(),
            conv.has_stride_w() ? conv.stride_w() : conv.stride(),

            conv.has_pad_h() ? conv.pad_h() : conv.pad(),
            conv.has_pad_w() ? conv.pad_w() : conv.pad(),

            weights[0]->shape.dims[dim_c] * conv.group(), weights[0]->shape.dims[dim_n],
            conv.group(), inputs, weights));
        return res;
    };

    /* deconv_2d */
    mp_parse_func_["Deconvolution"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              const ConvolutionParameter &conv = layerdef.convolution_param();
              auto inputs = gather_layer_input(layerdef);
              auto weights = gather_layer_param(layerweight);
              unique_ptr<fake_op> res(new fake_conv_2d(
                  conv.bias_term(), false, weights[0]->shape.dims[2], weights[0]->shape.dims[3],

                  conv.has_stride_h() ? conv.stride_h() : conv.stride(),
                  conv.has_stride_w() ? conv.stride_w() : conv.stride(),

                  conv.has_pad_h() ? conv.pad_h() : conv.pad(),
                  conv.has_pad_w() ? conv.pad_w() : conv.pad(),

                  weights[0]->shape.dims[1] * conv.group(), weights[0]->shape.dims[0], conv.group(),
                  inputs, weights));
              return res;
          };

    /* pool */
    mp_parse_func_["Pooling"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              const PoolingParameter &pool = layerdef.pooling_param();
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              unique_ptr<fake_op> res(new fake_pool(
                  pool.pool() == caffe::PoolingParameter::MAX ? fake_pool::pool_method::MAX
                      : pool.pool() == caffe::PoolingParameter::AVE
                      ? fake_pool::pool_method::AVE
                      : (abort(), fake_pool::pool_method::MAX),
                  pool.ceil_mode(),

                  pool.has_kernel_h() ? pool.kernel_h() : pool.kernel_size(),
                  pool.has_kernel_w() ? pool.kernel_w() : pool.kernel_size(),
                  pool.has_stride_h() ? pool.stride_h() : pool.stride(),
                  pool.has_stride_w() ? pool.stride_w() : pool.stride(),

                  pool.has_pad_h() ? pool.pad_h() : pool.pad(),
                  pool.has_pad_w() ? pool.pad_w() : pool.pad(), inputs));
              return res;
          };

    /* relu */
    mp_parse_func_["ReLU"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              unique_ptr<fake_op> res(new fake_relu(inputs));
              return res;
          };

    /* ip */
    mp_parse_func_["InnerProduct"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              const InnerProductParameter &ip = layerdef.inner_product_param();
              auto inputs = gather_layer_input(layerdef);
              auto weights = gather_layer_param(layerweight);
              if (!ip.bias_term() && 2 == weights.size()) {
                  weights.resize(1);
              }
              unique_ptr<fake_op> res(
                  new fake_ip(ip.bias_term(), false, ip.num_output(), inputs, weights));
              return res;
          };

    /* lrn */
    mp_parse_func_["LRN"] = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
        auto inputs = gather_layer_input(layerdef);
        (void)layerweight;
        const LRNParameter &lrn = layerdef.lrn_param();
        unique_ptr<fake_op> res(new fake_lrn(
            lrn.local_size(), lrn.alpha(), lrn.beta(), lrn.k(), lrn.norm_region(), inputs));
        return res;
    };

    /* eltwise */
    mp_parse_func_["Eltwise"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              const EltwiseParameter &eltwise = layerdef.eltwise_param();
              std::vector<float> coeff;
              for (int i = 0; i < eltwise.coeff_size(); ++i) {
                  coeff.push_back(eltwise.coeff(i));
              }
              unique_ptr<fake_op> res(new fake_eltwise(eltwise.operation(), coeff, inputs));
              return res;
          };

    /* BN */
    mp_parse_func_["BN"] = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
        auto inputs = gather_layer_input(layerdef);
        auto weights = gather_layer_param(layerweight);
        const BNParameter &bn = layerdef.bn_param();
        unique_ptr<fake_op> res(new fake_bn(bn.var_eps(), inputs, weights));
        return res;
    };

    /* softmax */
    mp_parse_func_["Softmax"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              const SoftmaxParameter &softmax = layerdef.softmax_param();
              unique_ptr<fake_op> res(new fake_softmax(softmax.axis(), inputs));
              return res;
          };

    /* concat */
    mp_parse_func_["Concat"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              const ConcatParameter &concat = layerdef.concat_param();
              unique_ptr<fake_op> res(new fake_concat(concat.axis(), inputs));
              return res;
          };

    /* sigmoid */
    mp_parse_func_["Sigmoid"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              (void)layerweight;
              unique_ptr<fake_op> res(new fake_sigmoid(inputs));
              return res;
          };

    /* reshape */
    mp_parse_func_["Reshape"] = [=](const LayerParameter &layerdef,
                                    const LayerParameter &layerweight) {
        auto inputs = gather_layer_input(layerdef);
        (void)layerweight;
        const ReshapeParameter &reshape = layerdef.reshape_param();
        std::vector<int32_t> dims;
        for (int i = 0; i < reshape.shape().dim_size(); ++i) {
            dims.push_back(reshape.shape().dim(i));
        }
        unique_ptr<fake_op> res(new fake_reshape(reshape.axis(), reshape.num_axes(), dims, inputs));
        return res;
    };

    /* interp */
    mp_parse_func_["Interp"] = [=](const LayerParameter &layerdef,
                                   const LayerParameter &layerweight) {
        auto inputs = gather_layer_input(layerdef);
        (void)layerweight;
        const InterpParameter &interp = layerdef.interp_param();
        uint32_t height, width, zoom_factor, shrink_factor, pad_beg, pad_end;
        if (interp.has_height() && interp.has_width()) {
            height = interp.height();
            width = interp.width();
        } else {
            height = 0;
            width = 0;
        }
        if (interp.has_zoom_factor()) {
            zoom_factor = interp.zoom_factor();
        } else {
            zoom_factor = 0;
        }
        if (interp.has_shrink_factor()) {
            shrink_factor = interp.shrink_factor();
        } else {
            shrink_factor = 0;
        }
        if (interp.has_pad_beg()) {
            pad_beg = interp.pad_beg();
        } else {
            pad_beg = 0;
        }
        if (interp.has_pad_end()) {
            pad_end = interp.pad_end();
        } else {
            pad_end = 0;
        }
        unique_ptr<fake_op> res(
            new fake_interp(height, width, zoom_factor, shrink_factor, pad_beg, pad_end, inputs));
        return res;
    };

    /* prelu */
    mp_parse_func_["PReLU"]
        = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
              auto inputs = gather_layer_input(layerdef);
              auto weights = gather_layer_param(layerweight);
              const PReLUParameter &prelu = layerdef.prelu_param();
              unique_ptr<fake_op> res(new fake_prelu(prelu.channel_shared(), inputs, weights));
              return res;
          };

    /* pad */
    mp_parse_func_["Pad"] = [=](const LayerParameter &layerdef, const LayerParameter &layerweight) {
        auto inputs = gather_layer_input(layerdef);
        const PadParameter &pad = layerdef.pad_param();
        std::vector<int32_t> pads;
        (void)layerweight;
        for (int i = 0; i < 8; ++i) {
            pads.push_back(pad.pads(i));
        }
        unique_ptr<fake_op> res(new fake_pad(pad.mode(), pad.value(), pads, inputs));
        return res;
    };
}

caffe_parser::caffe_parser(
    const NetParameter &netdef, const NetParameter &netweight,
    const std::vector<std::shared_ptr<fake_tensor>> inputs)
    : caffe_parser(netdef, netweight)
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i]->dtype() == dtFLOAT32)
            this->mp_tensor_[inputs[i]->name] = inputs[i];
    }
}
