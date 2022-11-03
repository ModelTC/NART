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
#include <map>
#include <sstream>

#include "include/fake.h"
#include "include/float_type.h"
#include "include/version.h"

#include "pybind11/cast.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "src/serialize.h"

#include "proxy.h"
#include "python_fakes.h"
#include "symbol.h"

namespace py = pybind11;
using namespace std;

namespace nart __attribute__((visibility("hidden"))) { }

struct create_tensor {
    template <typename T>
    static std::shared_ptr<nart::fake_tensor>
    get(vector<int> &shape, int channel_axis, int batch_axis, py::array *data)
    {
        shared_ptr<nart::fake_tensor_tp<T>> res
            = dynamic_pointer_cast<nart::fake_tensor_tp<T>>(nart::new_fake_tensor<T>());
        res->shape.channel_axis = channel_axis;
        res->shape.batch_axis = batch_axis;
        res->shape.dims = shape;
        if (nullptr != data) {
            res->v_.resize([&]() {
                size_t res = 1;
                for (auto i : shape) {
                    res *= i;
                }
                return res;
            }());
            memcpy(res->v_.data(), data->data(0), sizeof(T) * res->v_.size());
        }
        return res;
    }
};

PYBIND11_MODULE(_nart, m)
{
    m.doc() = "nart python interface";
    m.def("infer_shape", [](py::object obj) {
        auto o = nart::parse_op_from_py(obj);
        auto res = o->get_output_tensors();
        delete o;
        return res;
    });
    m.attr("numpy") = py::module::import("numpy");
    py::class_<nart::fake_tensor, std::shared_ptr<nart::fake_tensor>>(m, "FakeTensor")
        .def(
            "dtype",
            [](nart::fake_tensor &tensor) {
                return nart::fake_tensor::dtype_tp2str(tensor.dtype());
            })
        .def(
            "data",
            [m](nart::fake_tensor &tensor) -> py::object {
                if (tensor.data_size()) {
                    switch (tensor.dtype()) {
                    case dtFLOAT32:
                        return py::array(py::dtype("float32"), tensor.shape.dims, tensor.data());
                    // return m.attr("numpy").attr("ndarray")(tensor.shape.dims, "float32");
                    case dtFLOAT16:
                        return py::array(py::dtype("float16"), tensor.shape.dims, tensor.data());
                    // return m.attr("numpy").attr("ndarray")(tensor.shape.dims, "float16");
                    case dtUINT8:
                        return py::array(py::dtype("uint8"), tensor.shape.dims, tensor.data());
                    case dtINT8:
                        return py::array(py::dtype("int8"), tensor.shape.dims, tensor.data());
                    case dtUINT16:
                        return py::array(py::dtype("uint16"), tensor.shape.dims, tensor.data());
                    case dtINT16:
                        return py::array(py::dtype("int16"), tensor.shape.dims, tensor.data());
                    case dtUINT32:
                        return py::array(py::dtype("uint32"), tensor.shape.dims, tensor.data());
                    case dtINT32:
                        return py::array(py::dtype("int32"), tensor.shape.dims, tensor.data());
                    case dtUINT64:
                        return py::array(py::dtype("uint64"), tensor.shape.dims, tensor.data());
                    case dtINT64:
                        return py::array(py::dtype("int64"), tensor.shape.dims, tensor.data());
                    default:
                        throw std::invalid_argument("Unmplemented Datatype");
                    }
                } else {
                    return py::none();
                }
            })
        .def_property_readonly(
            "shape_channel_axis",
            [](nart::fake_tensor &tensor) { return tensor.shape.channel_axis; })
        .def_property_readonly(
            "shape_batch_axis", [](nart::fake_tensor &tensor) { return tensor.shape.batch_axis; })
        .def_property_readonly(
            "shape_dims", [](nart::fake_tensor &tensor) { return tensor.shape.dims; })
        .def_readwrite("name", &nart::fake_tensor::name);
    m.def(
        "get_true_tensor",
        [](string dtype, vector<int> shape, int channel_axis, int batch_axis,
           py::array data) -> std::shared_ptr<nart::fake_tensor> {
            if ("float32" == dtype)
                return create_tensor::get<float>(shape, channel_axis, batch_axis, &data);
            if ("float16" == dtype)
                return create_tensor::get<float16_t>(shape, channel_axis, batch_axis, &data);
            if ("int8" == dtype)
                return create_tensor::get<int8_t>(shape, channel_axis, batch_axis, &data);
            if ("uint8" == dtype)
                return create_tensor::get<uint8_t>(shape, channel_axis, batch_axis, &data);
            if ("int16" == dtype)
                return create_tensor::get<int16_t>(shape, channel_axis, batch_axis, &data);
            if ("uint16" == dtype)
                return create_tensor::get<uint16_t>(shape, channel_axis, batch_axis, &data);
            if ("int32" == dtype)
                return create_tensor::get<int32_t>(shape, channel_axis, batch_axis, &data);
            if ("uint32" == dtype)
                return create_tensor::get<uint32_t>(shape, channel_axis, batch_axis, &data);
            if ("int64" == dtype)
                return create_tensor::get<int64_t>(shape, channel_axis, batch_axis, &data);
            if ("uint64" == dtype)
                return create_tensor::get<uint64_t>(shape, channel_axis, batch_axis, &data);
            return std::shared_ptr<nart::fake_tensor>();
        });
    m.def(
        "get_true_tensor",
        [](string dtype, vector<int> shape, int channel_axis,
           int batch_axis) -> std::shared_ptr<nart::fake_tensor> {
            if ("float32" == dtype)
                return create_tensor::get<float>(shape, channel_axis, batch_axis, nullptr);
            if ("float16" == dtype)
                return create_tensor::get<float16_t>(shape, channel_axis, batch_axis, nullptr);
            if ("int8" == dtype)
                return create_tensor::get<int8_t>(shape, channel_axis, batch_axis, nullptr);
            if ("uint8" == dtype)
                return create_tensor::get<uint8_t>(shape, channel_axis, batch_axis, nullptr);
            if ("int16" == dtype)
                return create_tensor::get<int16_t>(shape, channel_axis, batch_axis, nullptr);
            if ("uint16" == dtype)
                return create_tensor::get<uint16_t>(shape, channel_axis, batch_axis, nullptr);
            if ("int32" == dtype)
                return create_tensor::get<int32_t>(shape, channel_axis, batch_axis, nullptr);
            if ("uint32" == dtype)
                return create_tensor::get<uint32_t>(shape, channel_axis, batch_axis, nullptr);
            if ("int64" == dtype)
                return create_tensor::get<int64_t>(shape, channel_axis, batch_axis, nullptr);
            if ("uint64" == dtype)
                return create_tensor::get<uint64_t>(shape, channel_axis, batch_axis, nullptr);
            return std::shared_ptr<nart::fake_tensor>();
        });
    py::enum_<nart::fake_transformer::frame_type_enum>(
        m, "FrameTypeEnum", py::arithmetic(), "frame type enumeration")
        .value("RGB", nart::fake_transformer::frame_type_enum::RGB)
        .value("BGR", nart::fake_transformer::frame_type_enum::BGR)
        .value("NV12", nart::fake_transformer::frame_type_enum::NV12)
        .value("NV21", nart::fake_transformer::frame_type_enum::NV21);
    py::class_<nart::fake_transformer>(m, "FakeTransformer")
        .def(py::init<>())
        .def_readwrite("tensor_name", &nart::fake_transformer::tensor_name)
        .def_readwrite("operators", &nart::fake_transformer::operators)
        .def_readwrite("frame_type", &nart::fake_transformer::frame_type)
        .def_property(
            "means",
            [](nart::fake_transformer &trans) {
                auto res = nart::fake_transformer::trans_px2vec(trans.means);
                return res;
            },
            [](nart::fake_transformer &trans, std::vector<float> &means) {
                trans.means = nart::fake_transformer::trans_vec2px(means);
            })
        .def_property(
            "stds",
            [](nart::fake_transformer &trans) {
                auto res = nart::fake_transformer::trans_px2vec(trans.stds);
                return res;
            },
            [](nart::fake_transformer &trans, std::vector<float> &stds) {
                trans.stds = nart::fake_transformer::trans_vec2px(stds);
            })
        .def_readwrite("paddings", &nart::fake_transformer::paddings);
    py::class_<nart::fake_parade>(m, "FakeParade")
        .def(py::init<>())
        .def(
            "append",
            [](nart::fake_parade &parade, py::handle obj) {
                std::unique_ptr<nart::fake_op> op(nart::parse_op_from_py(obj));
                auto ptr = parade.append(std::move(op));
                return ptr->get_output_tensors();
            })
        .def("mark_as_output", &nart::fake_parade::mark_as_output)
        .def(
            "bind_transformer",
            [](nart::fake_parade &parade, std::string &tensor_name,
               nart::fake_transformer &transformer) {
                // TODO:
                //  CHECK
                parade.mutable_transform_params()[tensor_name] = transformer;
            })
        .def("serialize", [](nart::fake_parade &parade) {
            stringstream ss;
            serialize_v2(&parade, ss);
            string &&str = ss.str();
            py::bytes res(str.data(), str.length());
            return res;
        });
    m.def("serialize_v1", [](nart::fake_parade &parade) {
        stringstream ss;
        serialize_v1(&parade, ss);
        string &&str = ss.str();
        py::bytes res(str.data(), str.length());
        return res;
    });
    m.def("serialize_v2", [](nart::fake_parade &parade) {
        stringstream ss;
        serialize_v2(&parade, ss);
        string &&str = ss.str();
        py::bytes res(str.data(), str.length());
        return res;
    });
    py::class_<nart::fake_python_cls, std::shared_ptr<nart::fake_python_cls>>(m, "FakeOpInterface")
        .def(py::init([](py::handle self, std::vector<std::shared_ptr<nart::fake_tensor>> inputs) {
            auto res = std::make_shared<nart::fake_python_cls>();
            res->init(self, inputs);
            return res;
        }))
        .def(py::init([](py::handle self, std::vector<std::shared_ptr<nart::fake_tensor>> inputs,
                         std::vector<std::shared_ptr<nart::fake_tensor>> params) {
            auto res = std::make_shared<nart::fake_python_cls>();
            res->init(self, inputs, params);
            return res;
        }))
        .def_property_readonly("inputs", [](nart::fake_python_cls &self) { return self.inputs_; })
        .def_property_readonly("params", [](nart::fake_python_cls &self) { return self.params_; })
        .def(
            "_op_tp_code",
            []() -> uint64_t {
                throw std::runtime_error("");
                return 0;
            })
        .def("_infer_shape", [](py::handle, py::handle, py::handle) {
            throw std::runtime_error("infer_shape should be implemented");
        });
    nart::art::proxy::register_python_module(&m);
    nart::symbol::register_python_module(&m);

    m.attr("__version__") = pybind11::cast(nart_version);
}
namespace nart {

std::vector<std::pair<std::string, fake_tensor::shape_t>> fake_python_cls::infer_shape()
{
    std::vector<std::pair<std::string, fake_tensor::shape_t>> res;
    pybind11::object obj = self.attr("_infer_shape")();
    for (auto item : obj.cast<py::list>()) {
        std::string dtype = item.cast<py::tuple>()[0].cast<string>();

        fake_tensor::shape_t sp;
        sp.dims.clear();
        py::handle shape = item.cast<py::tuple>()[1];
        for (auto ii : shape.attr("dims")) {
            sp.dims.push_back(ii.cast<int>());
        }
        sp.channel_axis = shape.attr("channel_axis").cast<int>();
        sp.batch_axis = shape.attr("batch_axis").cast<int>();
        res.push_back(std::make_pair(dtype, sp));
    }
    return res;
}
}
