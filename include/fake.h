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

#include <functional>
#include <istream>
#include <map>
#include <memory>
#include <set>
#include <stdint.h>
#include <string>
#include <vector>

#include "art/op.h"
#include "art/transform.h"

namespace nart {

class fake_tensor;
class fake_op;
struct fake_transformer;

class fake_tensor {
public:
    std::string name;
    struct shape_t {
        std::vector<int32_t> dims = { 1, 1, 1, 1 };
        int32_t channel_axis = 1;
        int32_t batch_axis = 0;
    } shape;
    virtual size_t dtype_size() = 0;
    virtual uint32_t dtype() = 0;
    virtual size_t data_size() = 0;
    virtual void *data() = 0;
    virtual std::shared_ptr<fake_tensor> new_with_same_type() = 0;
    ~fake_tensor() = default;

    static fake_tensor *new_tensor_with_dtype(uint32_t dtype);
    static uint32_t dtype_str2tp(std::string);
    static std::string dtype_tp2str(uint32_t);

protected:
    fake_tensor() = default;
};

class fake_op {
public:
    class fake_setting {
    public:
        virtual std::string get_string() const = 0;
        virtual uint32_t get_dtype() const = 0;
        virtual void serialize(std::ostream &os) const = 0;
        virtual bool is_repeated() const = 0;
        virtual ~fake_setting() = default;
    };

    template <typename tp> class fake_setting_tp : public fake_setting {
    public:
        fake_setting_tp(const tp &value) : value_(value) { }
        virtual std::string get_string() const override;
        virtual uint32_t get_dtype() const override;
        virtual void serialize(std::ostream &os) const;
        virtual bool is_repeated() const override { return false; }

    public:
        tp value_;
    };

    template <typename tp> class fake_setting_repeat_tp : public fake_setting {
    public:
        fake_setting_repeat_tp(const std::vector<tp> &value) : value_(value) { }
        virtual std::string get_string() const override;
        virtual uint32_t get_dtype() const override;
        virtual void serialize(std::ostream &os) const override;
        virtual bool is_repeated() const override { return true; }

    private:
        std::vector<tp> value_;
    };
    /* op */
public:
    fake_op(const std::vector<std::shared_ptr<fake_tensor>> input_tensors);
    fake_op(
        const std::vector<std::shared_ptr<fake_tensor>> input_tensors,
        const std::vector<std::shared_ptr<fake_tensor>> param_tensors);
    fake_op() = delete;
    fake_op(const fake_op &op) = delete;
    virtual ~fake_op() = default;

    virtual std::vector<std::shared_ptr<fake_tensor>> get_input_tensors() const
    {
        return input_tensors_;
    }
    virtual std::vector<std::shared_ptr<fake_tensor>> get_output_tensors();
    virtual std::vector<std::shared_ptr<fake_tensor>> get_param_tensors() const
    {
        return param_tensors_;
    }
    virtual std::shared_ptr<fake_tensor> output_tensor(int index) final;
    virtual std::shared_ptr<fake_tensor> param_tensor(int index) const final;
    virtual void set_setting(uint32_t, std::unique_ptr<fake_setting> &&) final;
    virtual const std::map<uint32_t, std::unique_ptr<fake_setting>> &get_settings() const final
    {
        return settings_;
    }
    virtual uint64_t get_op_tp_code() const final;
    virtual uint32_t op_group() const final { return op_group_; }
    virtual void set_op_group(uint32_t group) final { op_group_ = group; }
    using dtype = uint32_t;

protected:
    std::vector<std::shared_ptr<fake_tensor>> output_tensors_;
    virtual uint64_t _get_op_tp_code() const = 0;
    // virtual void infer_output() final;
public:
    using inferer_type = std::function<std::vector<std::pair<dtype, fake_tensor::shape_t>>(
        const fake_op *, std::vector<fake_tensor::shape_t>)>;

protected:
    inferer_type inferer_;

private:
    const std::vector<std::shared_ptr<fake_tensor>> input_tensors_;
    const std::vector<std::shared_ptr<fake_tensor>> param_tensors_;
    std::map<uint32_t, std::unique_ptr<fake_setting>> settings_;
    uint32_t op_group_ = 0;
    bool output_infered_ = false;
};

struct fake_transformer {
    enum class frame_type_enum {
        RGB = 1,
        BGR = 2,
        NV12 = 3,
        NV21 = 4,
    };

    std::string tensor_name;
    uint32_t operators;
    int32_t frame_type;
    pixel_t means;
    pixel_t stds;
    pixel_t paddings;

    static pixel_t trans_vec2px(std::vector<float> &vec)
    {
        pixel_t res;
        if (vec.size() == 1) {
            res.r = vec[0];
            res.g = -1;
            res.b = -1;
        } else if (vec.size() == 3) {
            res.r = vec[0];
            res.g = vec[1];
            res.b = vec[2];
        } else {
            throw std::runtime_error(
                "Unknown pixel format with " + std::to_string(vec.size()) + " channels.");
        }
        return res;
    }
    static std::vector<float> trans_px2vec(pixel_t &px)
    {
        std::vector<float> res;
        if (px.g != -1 and px.g != -1) {
            res.push_back(px.r);
            res.push_back(px.g);
            res.push_back(px.b);
        } else {
            res.push_back(px.r);
        }
        return res;
    }
};

template <typename tp>
std::unique_ptr<fake_op::fake_setting> new_fake_setting_repeat(const std::vector<tp> &arg)
{
    return std::unique_ptr<fake_op::fake_setting>(new fake_op::fake_setting_repeat_tp<tp>(arg));
}

template <typename tp> class fake_tensor_tp : public fake_tensor {
public:
    virtual size_t dtype_size() override;
    virtual uint32_t dtype() override;
    virtual size_t data_size() override { return v_.size() * sizeof(tp); }
    virtual void *data() override { return v_.data(); }
    virtual std::shared_ptr<fake_tensor> new_with_same_type() override
    {
        auto res = std::static_pointer_cast<fake_tensor>(std::make_shared<fake_tensor_tp<tp>>());
        res->shape = shape;
        return res;
    }
    std::vector<tp> v_;
};

template <typename tp, typename... Args>
std::shared_ptr<fake_tensor> new_fake_tensor(Args &&...args)
{
    return std::static_pointer_cast<fake_tensor>(
        std::make_shared<fake_tensor_tp<tp>>(std::forward<Args>(args)...));
}

class fake_parade {
public:
    using vec_fake_tensor = std::vector<std::shared_ptr<fake_tensor>>;
    std::vector<std::shared_ptr<fake_tensor>> get_input_tensors() const;
    std::vector<std::shared_ptr<fake_tensor>> get_param_tensors() const;
    std::vector<std::shared_ptr<fake_tensor>> get_output_tensors() const;
    std::vector<std::shared_ptr<fake_tensor>> get_all_tensors() const;
    void mark_as_output(std::shared_ptr<fake_tensor>);
    template <typename tp, typename... Args> fake_op *append(Args &&...args)
    {
        std::unique_ptr<fake_op> op(new tp(std::forward<Args>(args)...));
        auto res = op.get();
        _append(std::move(op));
        return res;
    }
    fake_op *append(std::unique_ptr<fake_op> &&op)
    {
        auto res = op.get();
        _append(std::move(op));
        return res;
    }
    const std::vector<std::unique_ptr<fake_op>> &ops() const { return ops_; }
    std::vector<std::unique_ptr<fake_op>> &mutable_ops() { return ops_; }

    const std::map<std::string, fake_transformer> &transform_params() const
    {
        return transform_params_;
    }
    std::map<std::string, fake_transformer> &mutable_transform_params()
    {
        return transform_params_;
    }

private:
    void _append(std::unique_ptr<fake_op> &&op);
    std::vector<std::unique_ptr<fake_op>> ops_;
    std::set<std::shared_ptr<fake_tensor>> marked_output_;
    std::map<std::string, fake_transformer> transform_params_;
};

} /* namespace nart */
