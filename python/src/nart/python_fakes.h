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
#include <iostream>

#include "include/fakes.h"

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"

namespace nart __attribute__((visibility("hidden")))
{

    fake_op *parse_op_from_py(pybind11::handle obj);
    class fake_python_cls;

    class fake_python_op : public fake_op {
    public:
        fake_python_op(
            std::shared_ptr<fake_python_cls> impl,
            const std::vector<std::shared_ptr<fake_tensor>> input_tensors,
            const std::vector<std::shared_ptr<fake_tensor>> param_tensors);
        virtual uint64_t _get_op_tp_code() const override;
        inferer_type &inferer() { return inferer_; }
        std::function<void(fake_op *)> create_post_;

    private:
        std::shared_ptr<fake_python_cls> impl_;
    };

    class fake_python_cls : public std::enable_shared_from_this<fake_python_cls> {
    public:
        void
        init(pybind11::handle self, const std::vector<std::shared_ptr<fake_tensor>> input_tensors)
        {
            this->self = self;
            inputs_ = input_tensors;
        }
        void init(
            pybind11::handle self, const std::vector<std::shared_ptr<fake_tensor>> input_tensors,
            const std::vector<std::shared_ptr<fake_tensor>> param_tensors)
        {
            inputs_ = input_tensors;
            params_ = param_tensors;
            this->self = self;
        }
        uint64_t get_op_tp_code() const { return self.attr("_op_tp_code")().cast<uint64_t>(); }
        fake_python_op *generate()
        {
            auto res = new fake_python_op(shared_from_this(), inputs_, params_);
            if (create_post_) {
                create_post_(res);
            }
            if (proxy_infer) {
                res->inferer() = proxy_infer;
            }
            apply_setting(res);
            auto op_code = get_op_tp_code();
            res->set_op_group(op_code >> 32);
            return res;
        }
        void apply_setting(fake_op *op);
        std::vector<std::pair<std::string, fake_tensor::shape_t>> infer_shape();
        std::vector<std::shared_ptr<fake_tensor>> inputs_;
        std::vector<std::shared_ptr<fake_tensor>> params_;
        std::vector<std::shared_ptr<fake_tensor>> outputs;
        fake_op::inferer_type proxy_infer;
        std::function<void(fake_op *)> create_post_;

    private:
        pybind11::handle self;
    };

}
namespace pybind11 {
template class class_<nart::fake_python_cls, std::shared_ptr<nart::fake_python_cls>>;
}
