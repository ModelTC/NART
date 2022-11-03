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

#include <caffe.pb.h>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include "include/parser/parser.h"

namespace nart {

std::map<std::string, std::vector<int>> get_input(const caffe::NetParameter &netdef);
const caffe::LayerParameter *
find_weight(const std::string &name, const caffe::NetParameter &weight);

class caffe_parser : public parser {
public:
    caffe_parser(const caffe::NetParameter &netdef, const caffe::NetParameter &netweight);
    caffe_parser(
        const caffe::NetParameter &netdef, const caffe::NetParameter &netweight,
        const std::vector<std::shared_ptr<fake_tensor>> inputs);
    virtual std::unique_ptr<fake_parade> parse() override;
    virtual ~caffe_parser() = default;
    caffe_parser() = delete;

protected:
    virtual void op_post(fake_op *op) { (void)op; }
    /* map from type string to parse function */
    using parse_function = std::function<std::unique_ptr<fake_op>(
        const caffe::LayerParameter &, const caffe::LayerParameter &)>;
    std::map<std::string, parse_function> mp_parse_func_;

protected:
    caffe::NetParameter netdef_;
    caffe::NetParameter netweight_;
    std::map<std::string, std::shared_ptr<fake_tensor>> mp_tensor_;
    std::function<std::shared_ptr<fake_tensor>(const std::string &, const std::vector<int> &)>
        init_input_func_ = [](const std::string &name, const std::vector<int> &shape) {
            auto res = new_fake_tensor<float>();
            res->shape.dims = shape;
            res->name = name;
            return res;
        };
    std::function<std::vector<std::shared_ptr<fake_tensor>>(const caffe::LayerParameter &)>
        init_weight_func_ = [](const caffe::LayerParameter &layer) {
            std::vector<std::shared_ptr<fake_tensor>> res;
            for (int i = 0; i < layer.blobs_size(); ++i) {
                const caffe::BlobProto &blob = layer.blobs(i);
                auto tensor = std::make_shared<fake_tensor_tp<float>>();
                if (blob.has_shape()) {
                    tensor->shape.dims.clear();
                    for (int i = 0; i < blob.shape().dim_size(); ++i) {
                        tensor->shape.dims.push_back(blob.shape().dim(i));
                    }
                } else {
                    tensor->shape.dims
                        = { blob.num(), blob.channels(), blob.height(), blob.width() };
                }
                tensor->v_.resize(blob.data_size());
                memcpy(tensor->v_.data(), blob.data().data(), sizeof(float) * tensor->v_.size());
                res.push_back(tensor);
            }
            return res;
        };
};

}
