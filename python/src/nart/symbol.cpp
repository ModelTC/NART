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
#include <functional>
#include <map>
#include <signal.h>
#include <sstream>
#include <iostream>

#include "art/default/default_module.h"
#include "art/module.h"
#include "art/parade.h"
#include "art/serialize.h"

#include "include/float_type.h"

#include "pybind11/cast.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "src/serialize.h"

#include "symbol.h"

namespace py = pybind11;
using namespace std;

const module_t *__pynart_reg_default_module_tp = &default_module_tp;

namespace nart {
namespace symbol {
    /* TODO: load symbols from other libs */
    struct global_data_t {
        global_data_t()
        {
            handle = dlopen("libart.so", RTLD_LAZY | RTLD_LOCAL);
            if (nullptr == handle and nullptr == dlsym(nullptr, "cpu_mem_tp"))
                cerr << dlerror() << endl;
            else {
            }
        }
        void *lookup(std::string name)
        {
            auto res = dlsym(handle, name.c_str());
            if (nullptr == res)
                error_str = dlerror();
            return res;
        }
        ~global_data_t()
        {
            if (nullptr != handle)
                dlclose(handle);
        }
        void *handle = nullptr;
        string error_str;
    };
    unique_ptr<global_data_t> _g;
    void *lookup_symbol(std::string name) { return _g->lookup(name); }
    std::string error()
    {
        auto res = _g->error_str;
        _g->error_str = "";
        return res;
    }
    void register_python_module(py::module *module)
    {
        auto m = module->def_submodule("symbol");
        if (nullptr == _g) {
            _g = unique_ptr<global_data_t>(new global_data_t);
        }
        m.def(
            "lookup", [](string name) -> unsigned long { return (unsigned long)_g->lookup(name); });
        m.def("error", error);
    }
}
}
