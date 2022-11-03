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

#include <functional>
#include <iostream>
#include <sstream>

#include "include/fakes.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "python_fakes.h"

namespace py = pybind11;
namespace nart __attribute__((visibility("hidden")))
{
    using namespace std;

    using parse_func_t = std::function<fake_op *(py::handle obj)>;
    /* py impl*/
    uint64_t fake_python_op::_get_op_tp_code() const { return impl_->get_op_tp_code(); }
    namespace {
        static vector<shared_ptr<fake_tensor>> parse_py_tensor_list(py::list lst)
        {
            vector<shared_ptr<fake_tensor>> res;
            for (py::handle item : lst) {
                res.push_back(item.attr("true_obj").cast<shared_ptr<nart::fake_tensor>>());
            }
            return res;
        }
        /* cpp impl */
        /* useless */
        static std::map<string, parse_func_t> mp_parse_func = []() {
            std::map<string, parse_func_t> res;
            res["Conv2D"] = [](py::handle obj) -> nart::fake_op * {
                return new nart::fake_conv_2d(
                    obj.attr("bias").cast<bool>(), obj.attr("relu").cast<bool>(),
                    obj.attr("kernel_h").cast<int>(), obj.attr("kernel_w").cast<int>(),
                    obj.attr("stride_h").cast<int>(), obj.attr("stride_w").cast<int>(),
                    obj.attr("pad_h").cast<int>(), obj.attr("pad_w").cast<int>(),
                    obj.attr("channel_in").cast<int>(), obj.attr("channel_out").cast<int>(),
                    obj.attr("group").cast<int>(), parse_py_tensor_list(obj.attr("inputs")),
                    parse_py_tensor_list(obj.attr("params")));

                return nullptr;
            };
            return res;
        }();
    }

    fake_python_op::fake_python_op(
        std::shared_ptr<fake_python_cls> impl,
        const std::vector<std::shared_ptr<fake_tensor>> input_tensors,
        const std::vector<std::shared_ptr<fake_tensor>> param_tensors)
        : fake_op(input_tensors, param_tensors), impl_(impl)
    {
        inferer_ = [](const fake_op *op, std::vector<fake_tensor::shape_t>) {
            std::vector<std::pair<dtype, fake_tensor::shape_t>> res;

            for (auto item : ((const fake_python_op *)op)->impl_->infer_shape()) {
                res.push_back(std::pair<uint32_t, fake_tensor::shape_t>(
                    fake_tensor::dtype_str2tp(item.first), item.second));
            }
            return res;
        };
    }

    fake_op *parse_op_from_py(py::handle obj)
    {
        fake_op *res = nullptr;
        if (py::hasattr(obj, "_op_tp_code")) {
            res = obj.cast<std::shared_ptr<fake_python_cls>>()->generate();
            auto rr = res->get_output_tensors();
        } else {
            string name = obj.attr("__class__").attr("__name__").cast<string>();
            auto find = mp_parse_func.find(name);
            if (mp_parse_func.end() != find) {
                res = (find->second)(obj);
            } else {
                throw std::runtime_error("class type [" + name + "] nort supported!");
            }
        }
        return res;
    }

    static std::unique_ptr<fake_op::fake_setting> parse_setting(
        uint32_t dtype, bool repeat, py::handle obj)
    {
        using res_tp = std::unique_ptr<fake_op::fake_setting>;
        if (repeat) {
            /* repeat */
            switch (dtype) {
            case dtUINT8:
                return res_tp(
                    new fake_op::fake_setting_repeat_tp<uint8_t>(obj.cast<vector<uint8_t>>()));
            // case dtUINT16:
            //     return res_tp(new
            //     fake_op::fake_setting_repeat_tp<uint16_t>(obj.cast<vector<uint16_t>>()));
            case dtUINT32:
                return res_tp(
                    new fake_op::fake_setting_repeat_tp<uint32_t>(obj.cast<vector<uint32_t>>()));
            // case dtINT8:
            //     return res_tp(new
            //     fake_op::fake_setting_repeat_tp<int8_t>(obj.cast<vector<int8_t>>()));
            // case dtINT16:
            //     return res_tp(new
            //     fake_op::fake_setting_repeat_tp<int16_t>(obj.cast<vector<int16_t>>()));
            case dtINT32:
                return res_tp(
                    new fake_op::fake_setting_repeat_tp<int32_t>(obj.cast<vector<int32_t>>()));
            case dtFLOAT32:
                return res_tp(
                    new fake_op::fake_setting_repeat_tp<float>(obj.cast<vector<float>>()));
            case dtSTR:
                return res_tp(
                    new fake_op::fake_setting_repeat_tp<string>(obj.cast<vector<string>>()));
                // case dtFLOAT64:
                //     return res_tp(new
                //     fake_op::fake_setting_repeat_tp<double>(obj.cast<vector<double>>()));
            }
        } else {
            /* single */
            switch (dtype) {
            case dtBOOL:
                return res_tp(new fake_op::fake_setting_tp<bool>(obj.cast<bool>()));
            case dtUINT8:
                return res_tp(new fake_op::fake_setting_tp<uint8_t>(obj.cast<uint8_t>()));
            case dtUINT16:
                return res_tp(new fake_op::fake_setting_tp<uint16_t>(obj.cast<uint16_t>()));
            case dtUINT32:
                return res_tp(new fake_op::fake_setting_tp<uint32_t>(obj.cast<uint32_t>()));
            case dtINT8:
                return res_tp(new fake_op::fake_setting_tp<int8_t>(obj.cast<int8_t>()));
            case dtINT16:
                return res_tp(new fake_op::fake_setting_tp<int16_t>(obj.cast<int16_t>()));
            case dtINT32:
                return res_tp(new fake_op::fake_setting_tp<int32_t>(obj.cast<int32_t>()));
            case dtFLOAT32:
                return res_tp(new fake_op::fake_setting_tp<float>(obj.cast<float>()));
            case dtFLOAT64:
                return res_tp(new fake_op::fake_setting_tp<double>(obj.cast<double>()));
            case dtSTR:
                return res_tp(new fake_op::fake_setting_tp<string>(obj.cast<string>()));
            }
        }
        ostringstream ss;
        ss << "cannot parse setting type " << datatype_name_from_type(dtype) << "("
           << (repeat ? "repeat" : "single") << ")";
        throw std::runtime_error(ss.str());
    }

    void fake_python_cls::apply_setting(fake_op * op)
    {
        if (py::hasattr(self, "_settings")) {
            for (py::handle item : self.attr("_settings")) {
                /* tuple: item, dtype, repeat, obj */
                auto tu = item.cast<py::tuple>();
                op->set_setting(
                    tu[0].cast<uint32_t>(),
                    parse_setting(
                        nart::fake_tensor::dtype_str2tp(tu[1].cast<string>()), tu[2].cast<bool>(),
                        tu[3]));
            }
        }
    }

}
