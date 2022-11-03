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

#include <algorithm>
#include <iostream>
#include <set>

#include "art/serialize.h"
#include "art/src/serialize_impl.h"
#include "art/tensor.h"

#include "include/fake.h"
#include "include/float_type.h"

using namespace std;
namespace nart {

static void serialize_write_data(const void *data, size_t size, ostream &os)
{
    os.write((const char *)data, size);
}

static void serialize_write_int32(int32_t value, ostream &os)
{
    os.write((const char *)&value, sizeof(value));
}

static void serialize_write_int64(int64_t value, ostream &os)
{
    os.write((const char *)&value, sizeof(value));
}

static void serialize_write_string(const string &str, ostream &os)
{
    serialize_write_int32(str.length(), os);
    os.write(str.c_str(), str.length());
}

void serialize_write_shape(const fake_tensor::shape_t &shape, ostream &os)
{
    serialize_write_int32(shape.dims.size(), os);
    fake_tensor::shape_t sp = shape;
    serialize_write_data(sp.dims.data(), sizeof(int32_t) * sp.dims.size(), os);
    serialize_write_data(&sp.channel_axis, sizeof(int32_t), os);
    serialize_write_data(&sp.batch_axis, sizeof(int32_t), os);
}

void serialize_write_head(int version, const string &name, ostream &os)
{
    serialize_write_int32(version, os);
    nart::serialize_write_string(name, os);
}

void serialize_v1(fake_parade *parade, ostream &os)
{
    if (nullptr == parade)
        return;
    auto serialize_write_int32 = [&](int32_t value) { nart::serialize_write_int32(value, os); };
    auto serialize_write_int64 = [&](int64_t value) { nart::serialize_write_int64(value, os); };
    auto serialize_write_string = [&](const string &v) { nart::serialize_write_string(v, os); };
    auto serialize_write_shape
        = [&](const fake_tensor::shape_t &v) { nart::serialize_write_shape(v, os); };
    auto serialize_write_data
        = [&](const void *data, size_t size) { nart::serialize_write_data(data, size, os); };

    nart::serialize_write_head(1, string(), os);
    vector<std::shared_ptr<fake_tensor>> input_tensors = parade->get_input_tensors();
    vector<std::shared_ptr<fake_tensor>> output_tensors = parade->get_output_tensors();
    vector<std::shared_ptr<fake_tensor>> param_tensors = parade->get_param_tensors();
    vector<std::shared_ptr<fake_tensor>> all_tensors;
    /* init tensor block */
    [&]() {
        int tensor_count = parade->get_all_tensors().size();
        serialize_write_int32(tensor_count);
        /* input tensors */
        [&]() {
            int input_count = input_tensors.size();
            serialize_write_int32(input_count);
            for (int i = 0; i < input_count; ++i) {
                serialize_write_int32(input_tensors[i]->dtype());
                serialize_write_string(input_tensors[i]->name);
                serialize_write_shape(input_tensors[i]->shape);
                all_tensors.push_back(input_tensors[i]);
            }
        }();
        /* weight tensors */
        [&]() {
            int weight_count = param_tensors.size();
            serialize_write_int32(weight_count);
            for (int i = 0; i < weight_count; ++i) {
                auto shape_count = [](const std::vector<int32_t> &shape) {
                    size_t s = 1;
                    for (int32_t i : shape)
                        s *= i;
                    return s;
                };
                serialize_write_int32(param_tensors[i]->dtype());
                serialize_write_string(param_tensors[i]->name);
                serialize_write_shape(param_tensors[i]->shape);
                serialize_write_data(
                    param_tensors[i]->data(),
                    param_tensors[i]->dtype_size() * shape_count(param_tensors[i]->shape.dims));
                all_tensors.push_back(param_tensors[i]);
            }
        }();
    }();
    /* ops */
    [&]() {
        vector<std::shared_ptr<fake_tensor>> tensors;
        const vector<unique_ptr<fake_op>> &ops = parade->ops();
        int32_t op_count = ops.size();
        serialize_write_int32(op_count);
        for (int32_t i = 0; i < op_count; ++i) {
            fake_op *op = ops[i].get();
            serialize_write_int32(op->get_input_tensors().size() + op->get_param_tensors().size());
            serialize_write_int32(op->get_output_tensors().size());
            serialize_write_int64(op->get_op_tp_code());
            for (std::shared_ptr<fake_tensor> tensor : op->get_input_tensors()) {
                int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), tensor)
                    - all_tensors.begin();
                if (idx >= (int32_t)all_tensors.size())
                    abort();
                serialize_write_int32(idx);
            }
            for (std::shared_ptr<fake_tensor> tensor : op->get_param_tensors()) {
                int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), tensor)
                    - all_tensors.begin();
                if (idx >= (int32_t)all_tensors.size())
                    abort();
                serialize_write_int32(idx);
            }
            serialize_write_int32(op->get_settings().size());
            for (auto &item : op->get_settings()) {
                serialize_write_int32(item.second->is_repeated());
                serialize_write_int32(item.first);
                serialize_write_int32(item.second->get_dtype());
                item.second->serialize(os);
            }
            for (shared_ptr<fake_tensor> tensor : op->get_output_tensors()) {
                all_tensors.push_back(tensor);
            }
        }
    }();
    /* output tensors */
    [&]() {
        sort(
            output_tensors.begin(), output_tensors.end(),
            [&all_tensors](const shared_ptr<fake_tensor> &a, const shared_ptr<fake_tensor> &b) {
                int32_t idxa
                    = std::find(all_tensors.begin(), all_tensors.end(), a) - all_tensors.begin();
                int32_t idxb
                    = std::find(all_tensors.begin(), all_tensors.end(), b) - all_tensors.begin();
                return (idxa < idxb);
            });
        int output_count = output_tensors.size();
        serialize_write_int32(output_count);
        for (int i = 0; i < output_count; ++i) {
            int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), output_tensors[i])
                - all_tensors.begin();
            serialize_write_int32(idx);
            serialize_write_string(output_tensors[i]->name);
        }
    }();
    /* share */
    [&]() { serialize_write_int32(0); }();
}

void serialize_v2(fake_parade *parade, ostream &os)
{
    if (nullptr == parade)
        return;
    auto serialize_write_int32 = [&](int32_t value) { nart::serialize_write_int32(value, os); };
    auto serialize_write_int64 = [&](int64_t value) { nart::serialize_write_int64(value, os); };
    auto serialize_write_string = [&](const string &v) { nart::serialize_write_string(v, os); };
    auto serialize_write_shape
        = [&](const fake_tensor::shape_t &v) { nart::serialize_write_shape(v, os); };
    auto serialize_write_data
        = [&](const void *data, size_t size) { nart::serialize_write_data(data, size, os); };
    auto serialize_write_pixel
        = [&](const pixel_t &px) { nart::serialize_write_data((void *)&px, sizeof(pixel_t), os); };

    nart::serialize_write_head(2, string(), os);
    vector<std::shared_ptr<fake_tensor>> input_tensors = parade->get_input_tensors();
    vector<std::shared_ptr<fake_tensor>> output_tensors = parade->get_output_tensors();
    vector<std::shared_ptr<fake_tensor>> param_tensors = parade->get_param_tensors();
    vector<std::shared_ptr<fake_tensor>> all_tensors;
    std::map<std::string, fake_transformer> transformers = parade->transform_params();
    /* init tensor block */
    [&]() {
        int tensor_count = parade->get_all_tensors().size();
        serialize_write_int32(tensor_count);
        /* input tensors */
        [&]() {
            int input_count = input_tensors.size();
            serialize_write_int32(input_count);
            for (int i = 0; i < input_count; ++i) {
                serialize_write_int32(input_tensors[i]->dtype());
                serialize_write_string(input_tensors[i]->name);
                serialize_write_shape(input_tensors[i]->shape);
                auto itr = transformers.find(input_tensors[i]->name);
                if (itr != transformers.end()) {
                    serialize_write_int32(1);
                    serialize_write_int32(itr->second.operators);
                    serialize_write_int32(itr->second.frame_type);
                    serialize_write_pixel(itr->second.means);
                    serialize_write_pixel(itr->second.stds);
                    serialize_write_pixel(itr->second.paddings);
                } else {
                    serialize_write_int32(0);
                }
                all_tensors.push_back(input_tensors[i]);
            }
        }();
        /* weight tensors */
        [&]() {
            int weight_count = param_tensors.size();
            serialize_write_int32(weight_count);
            for (int i = 0; i < weight_count; ++i) {
                auto shape_count = [](const std::vector<int32_t> &shape) {
                    size_t s = 1;
                    for (int32_t i : shape)
                        s *= i;
                    return s;
                };
                serialize_write_int32(param_tensors[i]->dtype());
                serialize_write_string(param_tensors[i]->name);
                serialize_write_shape(param_tensors[i]->shape);
                serialize_write_data(
                    param_tensors[i]->data(),
                    param_tensors[i]->dtype_size() * shape_count(param_tensors[i]->shape.dims));
                all_tensors.push_back(param_tensors[i]);
            }
        }();
    }();
    /* ops */
    [&]() {
        vector<std::shared_ptr<fake_tensor>> tensors;
        const vector<unique_ptr<fake_op>> &ops = parade->ops();
        int32_t op_count = ops.size();
        serialize_write_int32(op_count);
        for (int32_t i = 0; i < op_count; ++i) {
            fake_op *op = ops[i].get();
            serialize_write_int32(op->get_input_tensors().size() + op->get_param_tensors().size());
            serialize_write_int32(op->get_output_tensors().size());
            serialize_write_int64(op->get_op_tp_code());
            for (std::shared_ptr<fake_tensor> tensor : op->get_input_tensors()) {
                int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), tensor)
                    - all_tensors.begin();
                if (idx >= (int32_t)all_tensors.size())
                    abort();
                serialize_write_int32(idx);
            }
            for (std::shared_ptr<fake_tensor> tensor : op->get_param_tensors()) {
                int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), tensor)
                    - all_tensors.begin();
                if (idx >= (int32_t)all_tensors.size())
                    abort();
                serialize_write_int32(idx);
            }
            serialize_write_int32(op->get_settings().size());
            for (auto &item : op->get_settings()) {
                serialize_write_int32(item.second->is_repeated());
                serialize_write_int32(item.first);
                serialize_write_int32(item.second->get_dtype());
                item.second->serialize(os);
            }
            for (shared_ptr<fake_tensor> tensor : op->get_output_tensors()) {
                all_tensors.push_back(tensor);
            }
        }
    }();
    /* output tensors */
    [&]() {
        sort(
            output_tensors.begin(), output_tensors.end(),
            [&all_tensors](const shared_ptr<fake_tensor> &a, const shared_ptr<fake_tensor> &b) {
                int32_t idxa
                    = std::find(all_tensors.begin(), all_tensors.end(), a) - all_tensors.begin();
                int32_t idxb
                    = std::find(all_tensors.begin(), all_tensors.end(), b) - all_tensors.begin();
                return (idxa < idxb);
            });
        int output_count = output_tensors.size();
        serialize_write_int32(output_count);
        for (int i = 0; i < output_count; ++i) {
            int32_t idx = std::find(all_tensors.begin(), all_tensors.end(), output_tensors[i])
                - all_tensors.begin();
            serialize_write_int32(idx);
            serialize_write_string(output_tensors[i]->name);
        }
    }();
    /* share */
    [&]() { serialize_write_int32(0); }();
}

/* setting */
template <typename tp> uint32_t fake_op::fake_setting_tp<tp>::get_dtype() const { return 0; }

template <typename tp> void fake_op::fake_setting_tp<tp>::serialize(ostream &os) const
{
    serialize_write_data(&value_, datatype_sizeof(get_dtype()), os);
}

/* bool */
template <> string fake_op::fake_setting_tp<bool>::get_string() const
{
    return "[bool: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<bool>::get_dtype() const { return dtBOOL; }

/* int32 */
template <> string fake_op::fake_setting_tp<int32_t>::get_string() const
{
    return "[int32_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<int32_t>::get_dtype() const { return dtINT32; }

/* uint32 */
template <> string fake_op::fake_setting_tp<uint32_t>::get_string() const
{
    return "[uint32_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<uint32_t>::get_dtype() const { return dtUINT32; }

/* float32_t */
template <> string fake_op::fake_setting_tp<float32_t>::get_string() const
{
    return "[float32: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<float32_t>::get_dtype() const { return dtFLOAT32; }

/* float64_t */
template <> string fake_op::fake_setting_tp<double>::get_string() const
{
    return "[float64: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<double>::get_dtype() const { return dtFLOAT64; }

/* int8 */
template <> string fake_op::fake_setting_tp<int8_t>::get_string() const
{
    return "[int8_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<int8_t>::get_dtype() const { return dtINT8; }

/* uint8 */
template <> string fake_op::fake_setting_tp<uint8_t>::get_string() const
{
    return "[uint8_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<uint8_t>::get_dtype() const { return dtUINT8; }

/* int16 */
template <> string fake_op::fake_setting_tp<int16_t>::get_string() const
{
    return "[int16_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<int16_t>::get_dtype() const { return dtINT16; }

/* uint16 */
template <> string fake_op::fake_setting_tp<uint16_t>::get_string() const
{
    return "[uint16_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<uint16_t>::get_dtype() const { return dtUINT16; }

/* uint64 */
template <> string fake_op::fake_setting_tp<uint64_t>::get_string() const
{
    return "[uint64_t: " + to_string(value_) + "]";
}

template <> uint32_t fake_op::fake_setting_tp<uint64_t>::get_dtype() const { return dtUINT64; }

/* string */
template <> string fake_op::fake_setting_tp<string>::get_string() const
{
    return "[string: \"" + value_ + "\"]";
}

template <> uint32_t fake_op::fake_setting_tp<string>::get_dtype() const { return dtSTR; }

template <> void fake_op::fake_setting_tp<string>::serialize(std::ostream &os) const
{
    serialize_write_string(value_, os);
}

/* repeat */
template <typename tp> uint32_t fake_op::fake_setting_repeat_tp<tp>::get_dtype() const { return 0; }

template <typename tp> void fake_op::fake_setting_repeat_tp<tp>::serialize(ostream &os) const
{
    serialize_write_int32(value_.size(), os);
    serialize_write_data(value_.data(), datatype_sizeof(get_dtype()) * value_.size(), os);
}

/* float32_t repeat */
template <> string fake_op::fake_setting_repeat_tp<float32_t>::get_string() const
{
    return "[float32 * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<float32_t>::get_dtype() const
{
    return dtFLOAT32;
}

/* float64_t repeat */
template <> string fake_op::fake_setting_repeat_tp<double>::get_string() const
{
    return "[double * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<double>::get_dtype() const
{
    return dtFLOAT64;
}

/* uint8 repeat */
template <> string fake_op::fake_setting_repeat_tp<uint8_t>::get_string() const
{
    return "[uint8 * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<uint8_t>::get_dtype() const { return dtUINT8; }

/* int8 repeat */
template <> string fake_op::fake_setting_repeat_tp<int8_t>::get_string() const
{
    return "[int8 * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<int8_t>::get_dtype() const { return dtINT8; }

template <> void fake_op::fake_setting_repeat_tp<uint8_t>::serialize(ostream &os) const
{
    serialize_write_int32(value_.size(), os);
    serialize_write_data(value_.data(), value_.size() * datatype_sizeof(get_dtype()), os);
}

/* int32 repeat */
template <> string fake_op::fake_setting_repeat_tp<int32_t>::get_string() const
{
    return "[int32 * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<int32_t>::get_dtype() const { return dtINT32; }

template <> void fake_op::fake_setting_repeat_tp<int32_t>::serialize(ostream &os) const
{
    serialize_write_int32(value_.size(), os);
    serialize_write_data(value_.data(), value_.size() * datatype_sizeof(get_dtype()), os);
}

/* uint32 repeat */
template <> string fake_op::fake_setting_repeat_tp<uint32_t>::get_string() const
{
    return "[uint32 * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<uint32_t>::get_dtype() const
{
    return dtUINT32;
}

template <> void fake_op::fake_setting_repeat_tp<uint32_t>::serialize(ostream &os) const
{
    serialize_write_int32(value_.size(), os);
    serialize_write_data(value_.data(), value_.size() * datatype_sizeof(get_dtype()), os);
}

/* string repeat */
template <> string fake_op::fake_setting_repeat_tp<string>::get_string() const
{
    return "[string * " + to_string(value_.size()) + "]";
}

template <> uint32_t fake_op::fake_setting_repeat_tp<string>::get_dtype() const { return dtSTR; }

template <> void fake_op::fake_setting_repeat_tp<string>::serialize(ostream &os) const
{
    serialize_write_int32(value_.size(), os);
    for (const auto &v : value_) {
        serialize_write_string(v, os);
    }
}

template class fake_op::fake_setting_tp<bool>;
template class fake_op::fake_setting_tp<int8_t>;
template class fake_op::fake_setting_tp<uint8_t>;
template class fake_op::fake_setting_tp<int16_t>;
template class fake_op::fake_setting_tp<uint16_t>;
template class fake_op::fake_setting_tp<int32_t>;
template class fake_op::fake_setting_tp<uint32_t>;
// static_assert(sizeof(float16_t) == 2, "float16 not supported");
template class fake_op::fake_setting_tp<float32_t>;
template class fake_op::fake_setting_tp<double>;
template class fake_op::fake_setting_tp<uint64_t>;
template class fake_op::fake_setting_tp<string>;

template class fake_op::fake_setting_repeat_tp<int32_t>;
template class fake_op::fake_setting_repeat_tp<uint32_t>;
template class fake_op::fake_setting_repeat_tp<int8_t>;
template class fake_op::fake_setting_repeat_tp<uint8_t>;
template class fake_op::fake_setting_repeat_tp<float32_t>;

}
