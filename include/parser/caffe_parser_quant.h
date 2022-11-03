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

#include <caffe.pb.h>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include "include/parser/caffe_parser.h"
#include "include/parser/parser.h"

namespace nart {

class quant_param {
public:
    float scale;
    int zero_point;
    int bits;
    int qtype;
};

class caffe_parser_quant : public caffe_parser {
public:
    caffe_parser_quant(
        const caffe::NetParameter &netdef, const caffe::NetParameter &netweight,
        const std::map<std::string, quant_param> &q_param_map);
    caffe_parser_quant(
        const caffe::NetParameter &netdef, const caffe::NetParameter &netweight,
        const std::vector<std::shared_ptr<fake_tensor>> inputs,
        const std::map<std::string, quant_param> &q_param_map);
    virtual ~caffe_parser_quant() = default;
    caffe_parser_quant() = delete;
    virtual std::unique_ptr<fake_parade> parse() override;

protected:
    virtual void op_post(fake_op *op, const caffe::LayerParameter &layer);
    /* map from type string to parse function */
protected:
    std::map<std::string, quant_param> q_param_map_;
};

}
