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

#include <cstdlib>
#include <iostream>
#include <set>
#include <signal.h>

#include "art/data_type.h"
#include "art/log.h"

#include "include/fake.h"
#include "include/float_type.h"

using namespace nart;
using namespace std;

fake_tensor *fake_tensor::new_tensor_with_dtype(uint32_t dtype)
{
    switch (dtype) {
    case dtUINT8:
        return new fake_tensor_tp<uint8_t>();
    case dtINT8:
        return new fake_tensor_tp<int8_t>();
    case dtUINT16:
        return new fake_tensor_tp<uint16_t>();
    case dtINT16:
        return new fake_tensor_tp<int16_t>();
    case dtUINT32:
        return new fake_tensor_tp<uint32_t>();
    case dtINT32:
        return new fake_tensor_tp<int32_t>();
    case dtUINT64:
        return new fake_tensor_tp<uint64_t>();
    case dtINT64:
        return new fake_tensor_tp<int64_t>();
    case dtFLOAT32:
        return new fake_tensor_tp<float>();
    case dtFLOAT16:
        return new fake_tensor_tp<float16_t>();
    default:
        return nullptr;
    }
}

uint32_t fake_tensor::dtype_str2tp(std::string dtype)
{
    if (dtype == "int8")
        return dtINT8;
    if (dtype == "int16")
        return dtINT16;
    if (dtype == "int32")
        return dtINT32;
    if (dtype == "int64")
        return dtINT64;

    if (dtype == "uint8")
        return dtUINT8;
    if (dtype == "uint16")
        return dtUINT16;
    if (dtype == "uint32")
        return dtUINT32;
    if (dtype == "uint64")
        return dtUINT64;

    if (dtype == "float64")
        return dtFLOAT64;
    if (dtype == "float32")
        return dtFLOAT32;
    if (dtype == "float16")
        return dtFLOAT16;
    if (dtype == "string")
        return dtSTR;
    if (dtype == "str")
        return dtSTR;
    if (dtype == "bool")
        return dtBOOL;
    throw std::runtime_error("cannot convert dtype string " + dtype + " to dtype");
}

std::string fake_tensor::dtype_tp2str(uint32_t dtype)
{
    switch (dtype) {
    case dtUINT8:
        return "uint8";
    case dtINT8:
        return "int8";
    case dtUINT16:
        return "uint16";
    case dtINT16:
        return "int16";
    case dtUINT32:
        return "uint32";
    case dtINT32:
        return "int32";
    case dtUINT64:
        return "uint64";
    case dtINT64:
        return "int64";
    case dtFLOAT32:
        return "float32";
    case dtFLOAT16:
        return "float16";
    case dtSTR:
        return "string";
    case dtBOOL:
        return "bool";
    }
    throw std::runtime_error(
        "cannot convert dtype tp " + string(datatype_name_from_type(dtype)) + " to dtype");
}

fake_op::fake_op(const vector<std::shared_ptr<fake_tensor>> input_tensors)
    : input_tensors_(input_tensors)
{
    if (0 != input_tensors.size()) {
        uint32_t dtype = input_tensors[0]->dtype();
        inferer_ = [dtype](const fake_op *, vector<fake_tensor::shape_t> in) {
            vector<pair<fake_op::dtype, fake_tensor::shape_t>> res;
            for (const auto &item : in) {
                res.push_back(make_pair(dtype, item));
                break;
            }
            return res;
        };
    }
}

fake_op::fake_op(
    const vector<std::shared_ptr<fake_tensor>> input_tensors,
    const vector<std::shared_ptr<fake_tensor>> param_tensors)
    : input_tensors_(input_tensors), param_tensors_(param_tensors)
{
    if (0 != input_tensors.size()) {
        uint32_t dtype = input_tensors[0]->dtype();
        inferer_ = [dtype](const fake_op *, vector<fake_tensor::shape_t> in) {
            vector<pair<fake_op::dtype, fake_tensor::shape_t>> res;
            for (const auto &item : in) {
                res.push_back(make_pair(dtype, item));
                break;
            }
            return res;
        };
    }
}

void fake_op::set_setting(uint32_t item, std::unique_ptr<fake_setting> &&setting)
{
    settings_[item] = move(setting);
}

std::vector<std::shared_ptr<fake_tensor>> fake_op::get_output_tensors()
{
    if (false == output_infered_) {
        vector<fake_tensor::shape_t> input_shapes;
        for (const auto &input : get_input_tensors()) {
            input_shapes.push_back(input->shape);
        }
        for (const auto &weight : get_param_tensors()) {
            input_shapes.push_back(weight->shape);
        }
        vector<pair<dtype, fake_tensor::shape_t>> output_infos = inferer_(this, input_shapes);
        for (const auto &item : output_infos) {
            shared_ptr<fake_tensor> tensor(fake_tensor::new_tensor_with_dtype(item.first));
            tensor->shape = item.second;
            output_tensors_.push_back(tensor);
        }
        output_infered_ = true;
    }
    return output_tensors_;
}

std::shared_ptr<fake_tensor> fake_op::output_tensor(int index)
{
    auto o = get_output_tensors();
    if (index < 0 || index >= (int)o.size())
        return nullptr;
    return o[index];
}

uint64_t fake_op::get_op_tp_code() const
{
    uint64_t code = _get_op_tp_code();
    if (code & 0x80000000)
        return code;
    else
        return ((code & 0xffffffff) | ((uint64_t)op_group_ << 32));
}

std::shared_ptr<fake_tensor> fake_op::param_tensor(int index) const
{
    auto o = get_param_tensors();
    if (index < 0 || index >= (int)o.size())
        return nullptr;
    return o[index];
}

void fake_parade::mark_as_output(shared_ptr<fake_tensor> tensor)
{
    if (marked_output_.end() == marked_output_.find(tensor)) {
        auto all = get_all_tensors();
        for (auto ts : all) {
            if (ts == tensor) {
                marked_output_.insert(ts);
                return;
            }
        }
    }
}

void fake_parade::_append(unique_ptr<fake_op> &&op)
{
    auto res = op->get_output_tensors();
    ops_.push_back(move(op));
}

vector<std::shared_ptr<fake_tensor>> fake_parade::get_input_tensors() const
{
    vector<std::shared_ptr<fake_tensor>> res;
    set<std::shared_ptr<fake_tensor>> s;
    set<std::shared_ptr<fake_tensor>> s_defined;
    for (const unique_ptr<fake_op> &op : ops_) {
        for (std::shared_ptr<fake_tensor> tensor : op->get_input_tensors()) {
            if (s.end() != s.find(tensor) || s_defined.end() != s_defined.find(tensor)
                || 0 != tensor->data_size()) {
                continue;
            }
            s.insert(tensor);
            res.push_back(tensor);
        }
        for (std::shared_ptr<fake_tensor> tensor : op->get_output_tensors()) {
            s_defined.insert(tensor);
        }
    }
    return res;
}

vector<std::shared_ptr<fake_tensor>> fake_parade::get_param_tensors() const
{
    vector<std::shared_ptr<fake_tensor>> res;
    set<std::shared_ptr<fake_tensor>> s;
    for (const unique_ptr<fake_op> &op : ops_) {
        for (std::shared_ptr<fake_tensor> tensor : op->get_param_tensors()) {
            if (s.end() != s.find(tensor))
                continue;
            s.insert(tensor);
            res.push_back(tensor);
        }

        for (std::shared_ptr<fake_tensor> tensor : op->get_input_tensors()) {
            if (0 == tensor->data_size() || s.end() != s.find(tensor))
                continue;
            s.insert(tensor);
            res.push_back(tensor);
        }
    }
    return res;
}

vector<std::shared_ptr<fake_tensor>> fake_parade::get_output_tensors() const
{
    vector<std::shared_ptr<fake_tensor>> res;
    set<std::shared_ptr<fake_tensor>> s;
    set<std::shared_ptr<fake_tensor>> s_used;
    for (auto iter = ops_.rbegin(); iter != ops_.rend(); ++iter) {
        const unique_ptr<fake_op> &op = *iter;
        auto oo = op->get_output_tensors();
        for (auto iter = oo.rbegin(); iter != oo.rend(); ++iter) {
            std::shared_ptr<fake_tensor> tensor = *iter;
            if (s.end() != s.find(tensor))
                continue;
            if (s_used.end() != s_used.find(tensor)
                && marked_output_.end() == marked_output_.find(tensor))
                continue;
            res.push_back(tensor);
            s.insert(tensor);
        }
        for (std::shared_ptr<fake_tensor> tensor : op->get_input_tensors())
            s_used.insert(tensor);
    }
    return vector<std::shared_ptr<fake_tensor>>(res.rbegin(), res.rend());
}

vector<std::shared_ptr<fake_tensor>> fake_parade::get_all_tensors() const
{
    vector<std::shared_ptr<fake_tensor>> res;
    for (auto tensor : get_input_tensors())
        res.push_back(tensor);
    for (auto tensor : get_param_tensors())
        res.push_back(tensor);
    for (const auto &op : ops_) {
        for (auto tensor : op->get_output_tensors())
            res.push_back(tensor);
    }
    return res;
}

namespace nart {

/* tensor */
template <typename tp> size_t fake_tensor_tp<tp>::dtype_size() { return sizeof(tp); }

template <typename tp> uint32_t fake_tensor_tp<tp>::dtype() { return 0; }

template <> uint32_t fake_tensor_tp<int8_t>::dtype() { return dtINT8; }
template <> uint32_t fake_tensor_tp<int16_t>::dtype() { return dtINT16; }
template <> uint32_t fake_tensor_tp<int32_t>::dtype() { return dtINT32; }
template <> uint32_t fake_tensor_tp<int64_t>::dtype() { return dtINT64; }

template <> uint32_t fake_tensor_tp<uint8_t>::dtype() { return dtUINT8; }
template <> uint32_t fake_tensor_tp<uint16_t>::dtype() { return dtUINT16; }
template <> uint32_t fake_tensor_tp<uint32_t>::dtype() { return dtUINT32; }
template <> uint32_t fake_tensor_tp<uint64_t>::dtype() { return dtUINT64; }

template <> uint32_t fake_tensor_tp<float16_t>::dtype() { return dtFLOAT16; }
template <> uint32_t fake_tensor_tp<float32_t>::dtype() { return dtFLOAT32; }
template <> uint32_t fake_tensor_tp<double>::dtype() { return dtFLOAT64; }

template class fake_tensor_tp<int8_t>;
template class fake_tensor_tp<int16_t>;
template class fake_tensor_tp<int32_t>;
template class fake_tensor_tp<uint8_t>;
template class fake_tensor_tp<uint16_t>;
template class fake_tensor_tp<float16_t>;
template class fake_tensor_tp<float32_t>;

}
