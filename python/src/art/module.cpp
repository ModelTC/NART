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

#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "art/default/default_module.h"
#include "art/parade.h"
#include "art/serialize.h"

#include "pybind11/pybind11.h"

#include "module.h"

namespace py = pybind11;
using std::string;
using std::unordered_map;
using std::vector;

namespace art {

// Don't touch, this is used to link default_module into pyart.
const module_t *__pyart_reg_default_module_tp = &default_module_tp;

// This global variable stores the directory path of nart.art,
// where libxxx_module.so will be placed.
static std::string package_dir;

void *find_sym(const char *sym, const char *lib = nullptr)
{
    void *ret = dlsym(nullptr, sym);
    if (ret == nullptr and lib != nullptr) {
        // TODO(hujian): any safer way to concat path in pre C++17 ?
        std::string library_path = package_dir + "/" + lib;
        void *h = dlopen(library_path.c_str(), RTLD_NOW);
        if (h == nullptr)
            std::cerr << dlerror() << std::endl;
        ret = dlsym(h, sym);
        if (ret == nullptr)
            std::cerr << dlerror() << std::endl;
    }
    return ret;
}

inline string module_path(string module)
{
    // return string("../install/lib/libart_module_")+module+".so";
    return string("libart_module_") + module + ".so";
}

inline string module_name(string module) { return module + "_module_tp"; }

/**
 * @brief A context that manages workspaces requried by a parade.
 */
class RuntimeContext {
public:
    RuntimeContext(const vector<string> &modules, const string &input_module);
    ~RuntimeContext();

    /**
     * @brief Returns the names of workspaces created from loaded modules.
     *  Users can use this method to check which workspaces are created
     * successfully.
     */
    vector<string> loaded_workspaces() const;
    workspace_t *const *workspace_data() const;
    const mem_tp *get_input_mem_tp() const;

private:
    vector<string> modules;
    vector<workspace_t *> workspaces;
    const mem_tp *input_mem_tp;

    // names of modules which are failed to load.
    static std::unordered_set<string> load_failed;
};

std::unordered_set<string> RuntimeContext::load_failed;

RuntimeContext::RuntimeContext(const vector<string> &modules, const string &input_module)
{
    input_mem_tp = nullptr;
    for (const std::string &module : modules) {
        if (this->load_failed.count(module)) {
            // already tried to load this module, but failed, skip now.
            continue;
        }
        module_t *mod
            = (module_t *)find_sym(module_name(module).c_str(), module_path(module).c_str());
        if (mod == nullptr) {
            std::cerr << "fail to load module " << module.c_str() << std::endl;
            this->load_failed.insert(module);
            continue;
        }
        this->modules.push_back(module);
        workspace_t *w = workspace_new(mod, nullptr);
        workspaces.push_back(w);
        if (module == input_module)
            input_mem_tp = workspace_memtype(w);
    }
    if (input_mem_tp == nullptr)
        input_mem_tp = workspace_memtype(workspaces[0]);
    workspaces.push_back(nullptr);
}

RuntimeContext::~RuntimeContext()
{
    for (workspace_t *&w : workspaces) {
        if (w != nullptr)
            workspace_delete(w);
    }
}

workspace_t *const *RuntimeContext::workspace_data() const { return workspaces.data(); }

const mem_tp *RuntimeContext::get_input_mem_tp() const { return input_mem_tp; }

vector<string> RuntimeContext::loaded_workspaces() const { return modules; }

parade_t *buf2model(struct buffer_t *buf, const RuntimeContext &ctx)
{
    deserialize_param_t params
        = { .workspaces = ctx.workspace_data(), .input_mem_tp = ctx.get_input_mem_tp() };
    parade_t *model = deserialize_parade(buf, &params);
    parade_prepare(model);
    nart_buffer_delete(buf);
    return model;
}

/**
 * @brief A wrapper class of art parade. This class servers to ease the usage of
 * parade in cpp.
 */
class Parade {
public:
    Parade(struct buffer_t *buf, const RuntimeContext &ctx);
    ~Parade();

    /**
     * @brief Reshape the current Parade.
     */
    bool reshape(const unordered_map<string, vector<uint32_t>> &shapes);

    /**
     * @brief Do parade preparation after reshape is done, initialization any
     * resources required by operators, and allocate memory spaces for tensors.
     */
    void prepare();

    /**
     * @brief Do inference .
     */
    void forward();

    /**
     * @brief Get a tensor_t by it's name.
     */
    tensor_t *get_tensor_by_name(const std::string &name);

    /**
     * @brief Get outputs .
     */
    bool get_output(size_t *output_count, tensor_array_t *outputs);

    const vector<string> &get_input_names() const { return this->input_names; }
    const vector<string> &get_output_names() const { return this->output_names; }

private:
    parade_t *raw_parade;
    unordered_map<string, tensor_t *> tensor_by_name;
    vector<string> input_names;
    vector<string> output_names;
};

Parade::Parade(struct buffer_t *buf, const RuntimeContext &ctx)
{
    this->raw_parade = buf2model(buf, ctx);
    parade_apply_reshape(this->raw_parade);
    this->prepare();
    // collect tensors
    size_t input_count = 0;
    tensor_array_t input_tensors;
    parade_get_input_tensors(raw_parade, &input_count, &input_tensors);
    for (size_t idx = 0; idx < input_count; ++idx) {
        tensor_t *tensor = input_tensors[idx];
        this->input_names.push_back(tensor_name(tensor));
        this->tensor_by_name.insert({ tensor_name(tensor), tensor });
    }

    size_t output_count = 0;
    tensor_array_t output_tensors;
    parade_get_output_tensors(raw_parade, &output_count, &output_tensors);
    for (size_t idx = 0; idx < output_count; ++idx) {
        tensor_t *tensor = output_tensors[idx];
        this->output_names.push_back(tensor_name(tensor));
        this->tensor_by_name.insert({ tensor_name(tensor), tensor });
    }
}

Parade::~Parade()
{
    parade_delete(this->raw_parade);
    this->raw_parade = nullptr;
}

bool Parade::reshape(const unordered_map<string, vector<uint32_t>> &shapes)
{
    size_t input_count;
    tensor_array_t c_inputs;
    parade_get_input_tensors(raw_parade, &input_count, &c_inputs);
    for (size_t idx = 0; idx < input_count; ++idx) {
        tensor_t *tensor = c_inputs[idx];
        if (shapes.count(std::string(tensor->name)) == 0) {
            continue;
        }
        const vector<uint32_t> &shape = shapes.at(tensor->name);
        if (shape.size() != static_cast<size_t>(tensor->shape.dim_size)) {
            // throw std::runtime_error("shape dim mismatch");
            return false;
        }
        for (size_t idx = 0; idx < shape.size(); ++idx) {
            tensor->shape.dim[idx] = shape[idx];
        }
    }
    return parade_apply_reshape(raw_parade);
}

void Parade::prepare() { parade_prepare(this->raw_parade); }

void Parade::forward() { parade_run(this->raw_parade); }

tensor_t *Parade::get_tensor_by_name(const std::string &name)
{
    if (tensor_by_name.find(name) != tensor_by_name.end())
        return tensor_by_name[name];
    else
        return nullptr;
}

bool Parade::get_output(size_t *output_count, tensor_array_t *outputs)
{
    return parade_get_output_tensors(this->raw_parade, output_count, outputs);
}

} // namespace art

unordered_map<int, string> DTYPE_NAME = {
    {dtINT8,     "int8"   },
    { dtINT16,   "int16"  },
    { dtINT32,   "int32"  },
    { dtINT64,   "int64"  },
    { dtUINT8,   "uint8"  },
    { dtUINT16,  "uint16" },
    { dtUINT32,  "uint32" },
    { dtUINT64,  "uint64" },
    { dtFLOAT16, "float16"},
    { dtFLOAT32, "float32"},
    { dtFLOAT64, "float64"},
    { dtBOOL,    "bool"   },
};

PYBIND11_MODULE(_art, m)
{
    m.attr("numpy") = py::module::import("numpy");
    using art::Parade;
    using art::RuntimeContext;
    py::class_<art::Parade>(m, "Parade", py::dynamic_attr())
        .def(py::init([](py::bytes buf, py::handle context) {
            std::string d = buf;
            std::istringstream is(buf);
            buffer_t *buf_warp = nart_buffer_new(4096);
            auto buf_read = [](size_t sz, void *data, void *stream) -> size_t {
                auto d = new char[sz];
                ((std::istream *)stream)->read(d, sz);
                memcpy(data, d, sz);
                delete[] d;
                return sz;
            };

            buffer_set_buffer_read_func(buf_warp, buf_read, &is);

            return std::unique_ptr<Parade>(new Parade(buf_warp, context.cast<RuntimeContext &>()));
        }))
        .def(
            "input_shapes",
            [m](py::handle self) {
                Parade &parade_wrap = self.cast<Parade &>();
                py::dict input_shapes;
                for (const string &name : parade_wrap.get_input_names()) {
                    py::list py_shape;
                    const shape_t &shape = parade_wrap.get_tensor_by_name(name)->shape;
                    for (size_t idx = 0; idx < shape.dim_size; idx++) {
                        py_shape.append(shape.dim[idx]);
                    }
                    input_shapes[name.c_str()] = py_shape;
                }
                return input_shapes;
            })
        .def(
            "input_dtypes",
            [m](py::handle self) {
                Parade &parade_wrap = self.cast<Parade &>();
                py::dict input_dtypes;
                for (const string &name : parade_wrap.get_input_names()) {
                    uint32_t dtype = parade_wrap.get_tensor_by_name(name)->dtype;
                    input_dtypes[name.c_str()] = DTYPE_NAME.at(dtype);
                }
                return input_dtypes;
            })
        .def(
            "output_shapes",
            [m](py::handle self) {
                Parade &parade_wrap = self.cast<Parade &>();
                py::dict output_shapes;
                for (const string &name : parade_wrap.get_output_names()) {
                    py::list py_shape;
                    const shape_t &shape = parade_wrap.get_tensor_by_name(name)->shape;
                    for (size_t idx = 0; idx < shape.dim_size; idx++) {
                        py_shape.append(shape.dim[idx]);
                    }
                    output_shapes[name.c_str()] = py_shape;
                }
                return output_shapes;
            })
        .def(
            "output_dtypes",
            [m](py::handle self) {
                Parade &parade_wrap = self.cast<Parade &>();
                py::dict output_dtypes;
                for (const string &name : parade_wrap.get_output_names()) {
                    uint32_t dtype = parade_wrap.get_tensor_by_name(name)->dtype;
                    output_dtypes[name.c_str()] = DTYPE_NAME.at(dtype);
                }
                return output_dtypes;
            })
        .def(
            "get_tensor_shape",
            [m](py::handle self, std::string name) -> py::list {
                Parade &parade_wrap = self.cast<Parade &>();
                auto *tensor = parade_wrap.get_tensor_by_name(name);
                if (tensor == nullptr) {
                    return py::none();
                }
                py::list py_shape;
                const shape_t &shape = tensor->shape;
                for (size_t idx = 0; idx < shape.dim_size; idx++) {
                    py_shape.append(shape.dim[idx]);
                }
                return py_shape;
            })
        .def(
            "get_tensor_dtype",
            [m](py::handle self, std::string name) -> py::str {
                Parade &parade_wrap = self.cast<Parade &>();
                auto *tensor = parade_wrap.get_tensor_by_name(name);
                if (tensor == nullptr) {
                    return py::none();
                }
                uint32_t dtype = parade_wrap.get_tensor_by_name(name)->dtype;
                return py::str(DTYPE_NAME.at(dtype));
            })
        .def("forward", &Parade::forward)
        .def(
            "reshape",
            [m](py::handle self, py::dict shapes) {
                Parade &parade_wrap = self.cast<Parade &>();
                unordered_map<string, vector<uint32_t>> cpp_shapes;
                for (const auto &item : shapes) {
                    std::string name = item.first.cast<py::str>();
                    py::list shape = item.second.cast<py::list>();
                    std::vector<uint32_t> shape_vec;
                    for (const auto &dim : shape) {
                        shape_vec.push_back(dim.cast<uint32_t>());
                    }
                    cpp_shapes[name] = move(shape_vec);
                }
                bool ret = parade_wrap.reshape(cpp_shapes);
                if (!ret) {
                    return false;
                }
                parade_wrap.prepare();
                return true;
            })
        .def(
            "run",
            [m](py::handle self, py::dict &inputs, py::dict &outputs) -> bool {
                // this method accepts a dict of numpy.array, the do inference
                // with their content as tensor data. steps:
                // 1. check all inputs/outputs are given and their shape/dtype matches
                // the parade's input tensor.
                // 2. copy data to parade's input tensor's mem.
                // 3. do inference
                // 4. copy output tensors' data to ndarray in outputs.
                using std::string;
                using std::vector;
                Parade &parade = self.cast<Parade &>();

                const vector<string> &input_tensor_names = parade.get_input_names();
                for (const string &name : input_tensor_names) {
                    assert(inputs.contains(name) && "some input not given when calling parade.run");
                    tensor_t *tensor = parade.get_tensor_by_name(name);
                    py::array input = inputs[name.c_str()].cast<py::array>();
                    if (tensor->shape.dim_size != input.ndim()) {
                        throw py::value_error("input dim size not match");
                    }
                    auto np_shape = input.shape();
                    for (size_t idx = 0; idx < input.ndim(); idx++) {
                        if (tensor->shape.dim[idx] != np_shape[idx]) {
                            throw py::value_error("input shape not match");
                        }
                    }
                    if (DTYPE_NAME.at(tensor->dtype) != py::str(input.dtype()).cast<string>()) {
                        throw py::value_error("input dtype not match");
                    }
                }

                const vector<string> &output_tensor_names = parade.get_output_names();
                for (const string &name : output_tensor_names) {
                    assert(
                        outputs.contains(name) && "some output not given when calling parade.run");
                    tensor_t *tensor = parade.get_tensor_by_name(name);
                    py::array output = outputs[name.c_str()].cast<py::array>();
                    if (tensor->shape.dim_size != output.ndim()) {
                        throw py::value_error("output dim size not match");
                    }
                    auto np_shape = output.shape();
                    for (size_t idx = 0; idx < output.ndim(); idx++) {
                        if (tensor->shape.dim[idx] != np_shape[idx]) {
                            throw py::value_error("output shape not match");
                        }
                    }
                    if (DTYPE_NAME.at(tensor->dtype) != py::str(output.dtype()).cast<string>()) {
                        throw py::value_error("output dtype not match");
                    }
                }

                for (const string &name : input_tensor_names) {
                    py::array input = inputs[name.c_str()];
                    input = py::cast<py::array>(
                        py::module::import("numpy").attr("ascontiguousarray")(input));
                    tensor_t *tensor = parade.get_tensor_by_name(name);
                    memcpy(
                        mem_cpu_data(tensor->mem), input.data(),
                        datatype_sizeof(tensor->dtype) * shape_count(&tensor->shape));
                }
                parade.forward();
                tensor_array_t outputs_;
                size_t output_count;
                parade.get_output(&output_count, &outputs_);
                for (size_t i = 0; i < output_count; ++i) {
                    memcpy(
                        outputs.cast<py::dict>()[outputs_[i]->name]
                            .cast<py::array>()
                            .mutable_data(),
                        mem_cpu_data(outputs_[i]->mem),
                        datatype_sizeof(outputs_[i]->dtype) * shape_count(&outputs_[i]->shape));
                }

                return true;
            })
        .def(
            "get_raw_tensor_data",
            [m](py::handle self, py::str name) {
                // this method is used to get the raw pointer of certain tensor.
                Parade &parade_wrap = self.cast<Parade &>();
                tensor_t *tensor = parade_wrap.get_tensor_by_name(name.cast<string>());
                if (tensor == nullptr) {
                    return static_cast<intptr_t>(0);
                }
                auto data_ptr = mem_data(tensor->mem);
                return reinterpret_cast<intptr_t>(data_ptr);
            })
        .def(
            "get_input_names",
            [m](py::handle self) {
                Parade &parade_wrap = self.cast<Parade &>();
                py::list result;
                for (const auto &name : parade_wrap.get_input_names()) {
                    result.append(name);
                }
                return result;
            })
        .def("get_output_names", [m](py::handle self) {
            Parade &parade_wrap = self.cast<Parade &>();
            py::list result;
            for (const auto &name : parade_wrap.get_output_names()) {
                result.append(name);
            }
            return result;
        });

    py::class_<art::RuntimeContext>(m, "RuntimeContext")
        .def(py::init([](py::list modules, py::str input_module) {
            vector<string> module_names;
            for (auto module : modules) {
                module_names.push_back(module.cast<string>());
            }
            string input_module_name = input_module.cast<string>();
            return std::unique_ptr<art::RuntimeContext>(
                new art::RuntimeContext(module_names, input_module_name));
        }))
        .def_property_readonly("loaded_workspaces", [](py::handle self) {
            // returns names of created workspaces in this context.
            auto &ctx = self.cast<RuntimeContext &>();
            return ctx.loaded_workspaces();
        });

    m.def("set_package_dir", [](py::str package_path) { art::package_dir = package_path; });
}
