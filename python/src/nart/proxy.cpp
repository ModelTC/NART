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

#include "art/op_settings.h"

#include "include/fake.h"
#include "include/fakes.h"

#include "proxy.h"

#include "./python_fakes.h"
#include "./symbol.h"

namespace py = pybind11;
namespace nart {
namespace art {

    static py::object data_tr_optional(uint32_t dtype, uvalue_t v)
    {
        switch (dtype) {
        case dtUINT8:
            return py::cast(v.u8);
        case dtUINT16:
            return py::cast(v.u16);
        case dtUINT32:
            return py::cast(v.u32);
        case dtUINT64:
            return py::cast(v.u64);
        case dtINT8:
            return py::cast(v.i8);
        case dtINT16:
            return py::cast(v.i16);
        case dtINT32:
            return py::cast(v.i32);
        case dtINT64:
            return py::cast(v.i64);

        case dtFLOAT32:
            return py::cast(v.f32);

        case dtBOOL:
            return py::cast(v.b);

        case dtSTR:
            return py::cast(v.str);

        case dtCPTR:
            return py::cast(v.cptr);
        }
        throw std::runtime_error(
            std::string("cannot cast datatype ") + datatype_name_from_type(dtype));
    }

    /*
     * proxy of `module_t` in art, through which a list of supported op could be automaticly
     * constructed
     */
    class python_module_proxy {
    public:
        python_module_proxy(const module_t *module) : module_(module) { }
        uint32_t op_group() const { return module_->op_group; }
        op_tp_entry_t *search_op_tp_entry(std::string op_tp) const
        {
            for (op_tp_entry_t *p = module_->op_tp_entry; p && p->tp; ++p) {
                if (op_tp == p->tp->name)
                    return p;
            }
            return nullptr;
        }
        py::list op_tps() const
        {
            py::list res;
            for (op_tp_entry_t *p = module_->op_tp_entry; p && p->tp; ++p) {
                res.append(py::make_tuple(p->tp->op_tp_code, std::string(p->tp->name)));
            }
            return res;
        }
        py::list op_tp_constraints(std::string op_tp) const
        {
            auto p = search_op_tp_entry(op_tp);
            if (p->tp->name == op_tp) {
                /* list<(name:string, item:int, datatype:string, constraint:string [,
                 * default_value:object])>
                 */
                py::list res;
                for (const setting_constraint_t *pc = p->tp->constraints; pc->item != SETTING_END;
                     pc++) {
                    if (pc->ctp == ENUM_SETTING_CONSTRAINT_OPTIONAL) {
                        res.append(py::make_tuple(
                            pc->name, pc->item, datatype_name_from_type(pc->dtype), "optional",
                            data_tr_optional(pc->dtype, pc->constraint.optional.default_value)));
                    } else {
                        res.append(py::make_tuple(
                            pc->name, pc->item, datatype_name_from_type(pc->dtype),
                            [](enum_setting_constraint_tp_t ctp) {
                                switch (ctp) {
                                case ENUM_SETTING_CONSTRAINT_REQUIRED:
                                    return "required";
                                case ENUM_SETTING_CONSTRAINT_OPTIONAL:
                                    return "optional";
                                case ENUM_SETTING_CONSTRAINT_REPEATED:
                                    return "repeated";
                                default:
                                    return "error";
                                }
                            }(pc->ctp)));
                    }
                }
                return res;
            }
            throw std::runtime_error("cannot find op type '" + op_tp + "'");
        }

    private:
        const module_t *module_;
    };

    static std::unique_ptr<fake_op::fake_setting>
    parse_setting(const setting_constraint_t *constraint, py::handle obj)
    {
        using res_tp = std::unique_ptr<fake_op::fake_setting>;
        if (constraint->ctp == ENUM_SETTING_CONSTRAINT_REPEATED) {
            /* repeat */
            switch (constraint->dtype) {
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
            switch (constraint->dtype) {
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
        ss << "cannot parse setting type " << datatype_name_from_type(constraint->dtype) << "("
           << (constraint->ctp == ENUM_SETTING_CONSTRAINT_REPEATED ? "repeat" : "single") << ")";
        throw std::runtime_error(ss.str());
    }

    void proxy::register_python_module(py::module *module)
    {
        /* art.proxy */
        py::module &m = *module;
        py::class_<python_module_proxy>(m, "proxy")
            .def(py::init([](std::string name) {
                void *addr = nart::symbol::lookup_symbol(name + "_module_tp");
                if (nullptr == addr)
                    throw std::runtime_error("cannot find module '" + name + "'");
                return new python_module_proxy((const module_t *)addr);
            }))
            .def_property_readonly(
                "op_group", [](const python_module_proxy &self) { return self.op_group(); })
            .def_property_readonly(
                "op_tps", [](const python_module_proxy &self) { return self.op_tps(); })
            .def(
                "constraints_of",
                [](const python_module_proxy &self, std::string op_tp) {
                    return self.op_tp_constraints(op_tp);
                })
            .def_static("proxy_init", [](py::handle obj) {
                nart::fake_python_cls *true_obj = obj.cast<nart::fake_python_cls *>();
                auto proxy = obj.attr("module").attr("true_obj").cast<python_module_proxy *>();
                // auto dtype =
                // obj.attr("_inputs").cast<py::list>()[0].attr("dtype").cast<std::string>();
                std::string dtype = "float32";
                if (obj.attr("_inputs").cast<py::list>().size() > 0)
                    dtype
                        = obj.attr("_inputs").cast<py::list>()[0].attr("dtype").cast<std::string>();
                auto tp = proxy
                              ->search_op_tp_entry(
                                  obj.attr("__class__").attr("__name__").cast<std::string>())
                              ->tp;
                auto inferer = wrap_op_infer(tp, nart::fake_tensor::dtype_str2tp(dtype));
                true_obj->proxy_infer = inferer;
                true_obj->create_post_ = [obj, tp](fake_op *op) {
                    for (auto item : obj.attr("get_settings")()) {
                        uint32_t itm = item.cast<py::tuple>()[0].cast<std::uint32_t>();
                        for (const setting_constraint_t *p = tp->constraints;
                             p->item != SETTING_END; ++p) {
                            if (p->item == itm) {
                                auto set = parse_setting(p, item.cast<py::tuple>()[1]);
                                op->set_setting(itm, move(set));
                                break;
                            }
                        }
                    }
                };
            });
    }

}
}
