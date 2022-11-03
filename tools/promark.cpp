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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <unordered_map>

#include "art/default/default_module.h"
#include "art/log.h"
#include "art/module.h"
#include "art/parade.h"
#include "art/serialize.h"
#include "art/src/parade_impl.h"
#include "art/tensor.h"

#include "json11/json11.hpp"

#define TIO

#include "art/timer.h"

static struct {
    std::vector<workspace_t *> ws;
    std::vector<std::string> modules;
    const mem_tp *mem_type;
} g_ctx;

static size_t read_parade_bin(size_t sz, void *data, void *arg)
{
    CHECK_NE(arg, NULL);
    CHECK_NE(data, NULL);
    FILE *fp = (FILE *)arg;
    size_t cnt = fread(data, 1, sz, fp);
    return cnt;
}

parade_t *load_model(const char *parade_bin_path)
{
    FILE *fp = fopen(parade_bin_path, "rb");
    CHECK_NE(fp, NULL);
    struct buffer_t *buf = nart_buffer_new(0x8000);
    buffer_set_buffer_read_func(buf, read_parade_bin, (void *)fp);

    deserialize_param_t params = { .workspaces = g_ctx.ws.data(), .input_mem_tp = g_ctx.mem_type };
    parade_t *model = deserialize_parade(buf, &params);
    nart_buffer_delete(buf);
    fclose(fp);
    parade_prepare(model);
    return model;
}

std::string dtype2str(uint32_t tp)
{
    switch (tp) {
    case dtFLOAT32:
        return "FLOAT32";
    case dtFLOAT16:
        return "FLOAT16";
    case dtINT32:
        return "INT32";
    case dtINT16:
        return "INT16";
    case dtINT8:
        return "INT8";
    case dtUINT32:
        return "UINT32";
    case dtUINT16:
        return "UINT16";
    case dtUINT8:
        return "UINT8";
    default:
        return "UNKNOWN";
    }
}

void load_input(const char *input_bin_path, tensor_t *p)
{
    uint8_t *data = (uint8_t *)mem_cpu_data(p->mem);
    const size_t size = shape_count(&p->shape) * datatype_sizeof(p->dtype);
    FILE *fp = fopen(input_bin_path, "rb");
    CHECK_NE(fp, NULL);
    const size_t input_size = fread(data, 1, size, fp);
    fclose(fp);
    CHECK_EQ(size, input_size);
    (void)mem_cpu_data(p->mem); // flush cache
}

void save_output(const char *input_bin_path, tensor_t *p)
{
    uint8_t *data = (uint8_t *)mem_cpu_data(p->mem);
    const size_t size = shape_count(&p->shape) * datatype_sizeof(p->dtype);
    FILE *fp = fopen(input_bin_path, "wb");
    CHECK_NE(fp, NULL);
    const size_t input_size = fwrite(data, 1, size, fp);
    fclose(fp);
    CHECK_EQ(size, input_size);
    (void)mem_cpu_data(p->mem); // flush cache
}

void *find_sym(const char *sym, const char *lib = nullptr)
{
    void *ret = dlsym(nullptr, sym);
    if (ret == nullptr and lib != nullptr) {
        void *h = dlopen(lib, RTLD_NOW);
        if (h == nullptr)
            std::cerr << dlerror() << std::endl;
        ret = dlsym(h, sym);
        if (ret == nullptr)
            std::cerr << dlerror() << std::endl;
    }
    return ret;
}

bool init_modules(std::string config)
{
    std::string err;
    auto json = json11::Json::parse(config, err);
    if (!err.empty()) {
        std::cerr << "Parse config error: " << err << std::endl;
        return false;
    }
    auto json_ws = json["workspaces"];
    if (!json_ws.is_object()) {
        std::cerr << "Parse config error: "
                  << "[workspaces] required, but get: " << json_ws.dump() << std::endl;
        return false;
    }
    auto ws_obj = json_ws.object_items();
    for (auto ws : ws_obj) {
        std::string sym_name = ws.first + "_module_tp";
        std::string lib_name = "libart_module_" + ws.first + ".so";
        module_t *mod_sym = (module_t *)find_sym(sym_name.c_str(), lib_name.c_str());

        if (mod_sym == nullptr) {
            std::string x = ws.first;
            std::transform(
                x.begin(), x.end(), x.begin(), [](unsigned char ch) { return std::tolower(ch); });
            std::cerr << "[" << ws.first
                      << "] module not found, recompile nart-case with MODULES=" << x << std::endl;
            return false;
        }

        workspace_t *w = workspace_new(mod_sym, nullptr);
        g_ctx.ws.push_back(w);
        g_ctx.modules.push_back(ws.first);
    }
    g_ctx.ws.push_back(nullptr);

    return [&]() {
        auto json_mem = json["input_workspace"];
        if (!json_mem.is_string()) {
            std::cerr << "\"input_workspace\" required." << std::endl;
        }
        std::string sym_name = json_mem.string_value();
        for (int i = 0; i < g_ctx.modules.size(); ++i) {
            if (g_ctx.modules[i] == sym_name) {
                g_ctx.mem_type = workspace_memtype(g_ctx.ws[i]);
                return true;
            }
        }
        std::cerr << "required input workspace [" << sym_name << "] has not been registered."
                  << std::endl;
        return false;
    }();
}

std::vector<std::string> split(const std::string &str, char delim)
{
    std::vector<std::string> ret;
    std::string temp;
    for (char c : str) {
        if (c != delim) {
            temp += c;
        } else {
            if (temp.size() != 0) {
                ret.emplace_back(std::move(temp));
            }
        }
    }
    if (temp.size() != 0) {
        ret.emplace_back(std::move(temp));
    }
    return ret;
}

int main(int argc, char *argv[])
{
    /* parse args */
    auto print_helper = [&]() -> int {
        std::cerr
            << "Usage: " << std::endl
            << argv[0]
            << " -m model [-c config] [-n run_times] [{-i blob.bin}] [-s] [-d] [-b batch_size]"
            << std::endl;
        return 1;
    };
    if (argc == 1) {
        return print_helper();
    }

    std::string model_file;
    std::string config_file;
    int run_times = 10;
    std::vector<std::string> inputs;
    bool save_output_flag = false;
    bool randomized_data = false;
    // whether to incldue host-device copy when benchmarking.
    bool include_host_device_copy = false;
    int ch;
    int batch_size = -1;
    using std::string;
    using std::vector;
    std::unordered_map<string, vector<int>> shape_by_name;
    while (-1 != (ch = getopt(argc, argv, "i:sm:c:n:b:r:d;y"))) {
        switch (ch) {
        case 'm':
            model_file = optarg;
            break;
        case 'c': {
            if (optarg == nullptr)
                return print_helper();

            config_file = optarg;
            break;
        }
        case 'n': {
            if (optarg == nullptr)
                return print_helper();

            run_times = atoi(optarg);
            break;
        }
        case 'i': {
            if (nullptr != optarg)
                inputs.push_back(optarg);
            break;
        }
        case 'b': {
            if (nullptr != optarg)
                batch_size = atoi(optarg);
            break;
        }
        case 's':
            save_output_flag = true;
            break;
        case 'r': {
            if (optarg == nullptr)
                return print_helper();
            string arg(optarg);
            auto pos = arg.find(':', 0);
            if (pos == std::string::npos) {
                break;
            }
            string name = arg.substr(0, pos);
            auto items = split(arg.substr(pos + 1), ',');
            vector<int> shape;
            for (const auto &dim : items) {
                shape.push_back(std::stoi(dim));
            }
            shape_by_name.insert({ std::move(name), std::move(shape) });
            break;
        }
        case 'd': {
            randomized_data = true;
            break;
        }
        case 'y': {
            include_host_device_copy = true;
            break;
        }
        default:
            return print_helper();
        }
    }
    if (model_file.empty())
        return print_helper();

    std::cout << "parse args ..." << std::endl
              << "  modelfile  : " << model_file << std::endl
              << "  config_file: " << config_file << std::endl
              << "  run_time   : " << run_times << std::endl
              << std::endl;
    std::ifstream fin(config_file);
    std::string config;
    if (fin) {
        fin.seekg(0, std::ios::end);
        size_t sz = fin.tellg();
        fin.seekg(0, std::ios::beg);
        config.resize(sz);
        fin.read(&config[0], sz);
    } else {
        config = R"STR(
{
"workspaces": {
"default": {}
},
"input_workspace": "default"
}
)STR";
    }

    /* get ws */
    if (!init_modules(config)) {
        std::cerr << "init module failed!" << std::endl;
        return 1;
    }
    /* load model */
    auto parade = load_model(model_file.c_str());

    /* print model */
    size_t input_cnt;
    tensor_array_t input_tensors;
    CHECK(parade_get_input_tensors(parade, &input_cnt, &input_tensors));
    size_t output_cnt;
    tensor_array_t output_tensors;
    CHECK(parade_get_output_tensors(parade, &output_cnt, &output_tensors));

    int cnt = 0;
    int i, j;

    /* reshape */
    if (batch_size > 0) {
        for (i = 0; i < input_cnt; i++) {
            input_tensors[i]->shape.dim[input_tensors[i]->shape.batch_axis] = batch_size;
        }
        if (!parade_apply_reshape(parade)) {
            LOG_error("Reshape parade failed\n");
        }
    }
    if (shape_by_name.size() > 0) {
        for (size_t i = 0; i < input_cnt; i++) {
            if (shape_by_name.count(std::string(input_tensors[i]->name)) != 0) {
                const auto &shape = shape_by_name[input_tensors[i]->name];
                input_tensors[i]->shape.dim_size = shape.size();
                for (size_t idim = 0; idim < shape.size(); ++idim) {
                    input_tensors[i]->shape.dim[idim] = shape[idim];
                }
            }
        }
        parade_apply_reshape(parade);
        if (!parade_apply_reshape(parade)) {
            LOG_error("Reshape parade failed\n");
        }
    }

    printf("input tensors:\n");
    for (i = 0; i < input_cnt; i++) {
        printf("\t%s: ", input_tensors[i]->name);
        std::cout << " " << dtype2str(input_tensors[i]->dtype) << " ";
        shape_t *shape = &input_tensors[i]->shape;
        printf("%d", shape->dim[0]);
        for (j = 1; j < shape->dim_size; j++) {
            printf(", %d", shape->dim[j]);
        }
        printf("\n");
    }

    printf("output tensors:\n");
    for (i = 0; i < output_cnt; i++) {
        printf("\t%s: ", output_tensors[i]->name);
        std::cout << " " << dtype2str(output_tensors[i]->dtype) << " ";
        shape_t *shape = &output_tensors[i]->shape;
        printf("%d", shape->dim[0]);
        for (j = 1; j < shape->dim_size; j++) {
            printf(", %d", shape->dim[j]);
        }
        printf("\n");
    }

    /* print model */
    {
        constexpr int head_width = 20;
        constexpr int col_width = 35;
        constexpr int shape_width = 20;
        constexpr int total_width = head_width + col_width * 2 + 6;
        constexpr char col_sep = '|';
        constexpr char row_sep = '-';

        struct _parade_t *parade_impl = (struct _parade_t *)parade;
        auto p_tensor = [](tensor_t *tensor) {
            if (nullptr == tensor)
                return std::string();
            std::stringstream ss;
            for (size_t i = 0; i < tensor->shape.dim_size; ++i) {
                if (i != 0)
                    ss << ",";
                ss << tensor->shape.dim[i];
            }
            std::string shape = "(" + ss.str() + ")";
            std::stringstream sss;
            sss << std::setw(shape_width) << shape;
            std::string aligned_shape = sss.str();
            if (nullptr == tensor_name(tensor)) {
                char x[20];
                sprintf(&x[0], "%p", tensor);
                return std::string(x) + aligned_shape;
            }
            return std::string(tensor_name(tensor)) + aligned_shape;
        };

        std::cout << std::endl << "DETAILED INFOMATION:" << std::endl;
        std::cout << std::setfill('=') << std::setw(total_width) << "" << std::endl;
        std::cout << std::setfill(' ') << col_sep << col_sep << std::setw(head_width) << "type"
                  << col_sep << std::setw(col_width) << "input tensors" << col_sep
                  << std::setw(col_width) << "output tensors" << col_sep << col_sep << std::endl;
        for (int i = 0; i < parade_impl->op_count; ++i) {
            std::cout << std::setfill(row_sep) << std::setw(total_width) << "" << std::endl;
            op_t *op = parade_impl->ops[i];
            for (int j = 0; j < std::max(op->input_size, op->output_size); j++) {
                std::cout << std::setfill(' ') << col_sep << col_sep << std::setw(head_width)
                          << (j == 0 ? std::string(workspace_name(op->workspace)) + " "
                                      + op->entry->tp->name
                                     : "")
                          << col_sep << std::setw(col_width)
                          << (j < op->input_size ? p_tensor(op->input_tensors[j]) : "") << col_sep
                          << std::setw(col_width)
                          << (j < op->output_size ? p_tensor(&op->output_tensors[j]) : "")
                          << col_sep << col_sep << std::endl;
            }
        }
        std::cout << std::setfill('=') << std::setw(total_width) << "" << std::endl;
    }

    /* test */
    for (i = 0; i < input_cnt; ++i) {
        memset(
            mem_cpu_data(input_tensors[i]->mem), 0,
            shape_count(&input_tensors[i]->shape) * datatype_sizeof(input_tensors[i]->dtype));
    }
    std::cout << std::endl << "Warming up ..." << std::endl;
    parade_run(parade);
    parade_run(parade);
    std::cout << std::endl << "Finish warm up" << std::endl;

    struct timeval start, end;
    float total_cost = 0.;
    for (int i = 0; i < run_times; i++) {
        if (randomized_data) {
            printf("use randomized data.\n");
            for (j = 0; j < input_cnt; ++j) {
                size_t data_cnt = shape_count(&input_tensors[j]->shape);
                for (int k = 0; k < data_cnt; ++k) {
                    ((char *)mem_cpu_data(input_tensors[j]->mem))[k] = rand();
                }
            }
        }
        if (include_host_device_copy) {
            for (j = 0; j < input_cnt; ++j) {
                mem_cpu_data(input_tensors[j]->mem);
            }
            for (j = 0; j < output_cnt; ++j) {
                mem_data(output_tensors[j]->mem);
            }
        }

        gettimeofday(&start, NULL);
        parade_run(parade);
        if (include_host_device_copy) {
            for (j = 0; j < output_cnt; ++j) {
                mem_cpu_data(output_tensors[j]->mem);
            }
        }
        gettimeofday(&end, NULL);
        float tmp = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.;
        total_cost += tmp;
        std::cout << "Process " << i << " : cost " << tmp << "ms" << std::endl;
    }
    std::cout << std::endl
              << "Test finised, average cost: " << total_cost / run_times << "ms" << std::endl
              << std::endl;

    /* save output (optional) */
    if (save_output_flag) {
        if (inputs.size() != input_cnt) {
            std::cerr << "Warning: Inputs file mismatch with model's inputs, which requires "
                      << input_cnt << ", but given " << inputs.size() << std::endl;
        } else {
            for (i = 0; i < input_cnt; ++i) {
                load_input(inputs[i].c_str(), input_tensors[i]);
            }
        }
        parade_run(parade);
        std::cerr << "Info: Saving output tensors ..." << std::endl;
        for (i = 0; i < output_cnt; ++i) {
            std::string name = std::string(tensor_name(output_tensors[i]));
            std::transform(name.begin(), name.end(), name.begin(), [](char ch) {
                return ch == '/' ? '_' : ch;
            });
            std::string out_name = std::string("nart-out-") + name + ".bin";
            save_output(out_name.c_str(), output_tensors[i]);
        }
    }

    /* deinit */
    parade_delete(parade);
    for (int i = 0; i < g_ctx.ws.size() - 1; ++i) {
        workspace_delete(g_ctx.ws[i]);
    }
    return 0;
}
