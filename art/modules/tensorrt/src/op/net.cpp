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

#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/profiler.hpp"
#include "art/tensorrt/tensorrt_op_settings.h"
#include "art/tensorrt/tensorrt_op_tp.h"
#include "art/tensorrt/tensorrt_workspace.h"

#include "NvInfer.h"

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO && severity != Severity::kVERBOSE) {
            std::cerr << "[trt] " << msg << std::endl;
        }
    }
} gLogger;

class ProfileReceiver : public nvinfer1::IProfiler {
public:
    ProfileReceiver();

    void reportLayerTime(const char *layerName, float ms) noexcept override;

    static ProfileReceiver &get_instance()
    {
        static std::unique_ptr<ProfileReceiver> instance;
        if (instance == nullptr) {
            instance.reset(new ProfileReceiver());
        }
        return *instance;
    }

private:
    std::ofstream output_file;
};

ProfileReceiver::ProfileReceiver()
{
    const char *out_str = getenv("TRT_PROFILING_OUTPUT");
    std::string output { out_str != nullptr ? out_str : "" };
    if (output.empty() || output == "stdout") {
        // nothing to do
    } else {
        // treat output as a file path
        output_file.open(output, std::ios::out);
        if (!output_file.is_open()) {
            gLogger.log(
                nvinfer1::ILogger::Severity::kWARNING,
                "cannot open profiling output file, redirecting to stdout instead");
        }
    }
}

void ProfileReceiver::reportLayerTime(const char *layerName, float ms) noexcept
{
    std::ostream &outs = output_file.is_open() ? output_file : std::cout;
    outs << layerName << " " << ms << "ms\n";
}

typedef struct {
    nvinfer1::IRuntime *runtime_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
} nvinfer_;

namespace {
typedef struct {
    op_t o;
    nvinfer_ *nv_net;
    std::vector<std::string> outputs;
    std::vector<std::string> inputs;
    void **gpu_buf;
    bool is_explicit_batch;
} op_net_t;
}

extern "C" {
op_net_t *op_tensorrt_net_tp_alloc(workspace_t *ws);
void op_tensorrt_net_tp_config(op_t *op);
bool op_infer_shape_tensorrt_net(op_t *op);
void op_tensorrt_net_tp_prepare(op_t *op);
void op_tensorrt_net_tp_destroy(op_t *op);
void op_tensorrt_net_tp_dealloc(op_t *op);

} // extern "C"

op_net_t *op_tensorrt_net_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_net_t *res = new op_net_t;
    memset(res, 0, sizeof(op_t));
    res->nv_net = new nvinfer_;
    res->gpu_buf = nullptr;
    return res;
}

void op_tensorrt_net_tp_config(op_t *op)
{
    op_net_t *net = (op_net_t *)op;

    // load serized model
    size_t sz;
    uint8_t *buffer;
    op_setting_array_get(op, SETTING_TENSORRT_NET, dtUINT8, &sz, &buffer);
    net->nv_net->runtime_ = nvinfer1::createInferRuntime(gLogger);

    net->nv_net->engine_
        = net->nv_net->runtime_->deserializeCudaEngine((void *)buffer, sz, nullptr);
    net->nv_net->context_ = net->nv_net->engine_->createExecutionContext();

#ifdef TRT_PROFILING
    if (getenv("TRT_PROFILING_OUTPUT") != NULL) {
        net->nv_net->context_->setProfiler(&ProfileReceiver::get_instance());
    }
#endif

    // load input label nams
    char **ins;
    size_t in_size;
    op_setting_array_get(op, SETTING_TENSORRT_INPUTS, dtSTR, &in_size, &ins);
    CHECK_GT(in_size, 0);
    net->inputs.resize(0);
    for (size_t i = 0; i < in_size; ++i) {
        net->inputs.push_back(ins[i]);
    }

    // load output label name
    char **outs;
    size_t out_size;
    op_setting_array_get(op, SETTING_TENSORRT_OUTPUTS, dtSTR, &out_size, &outs);
    CHECK_GT(out_size, 0);
    net->outputs.resize(0);
    for (size_t i = 0; i < out_size; ++i) {
        net->outputs.push_back(outs[i]);
    }

    bool explicit_batch = !(net->nv_net->engine_->hasImplicitBatchDimension());
    net->is_explicit_batch = explicit_batch;

    if (!explicit_batch) {
        // check intput tensor batch size. in TensorRT, all inputs' batch size should be the same in
        // implicit batch mode.
        for (int i = 0; i < op->input_size - 1; ++i) {
            CHECK_EQ(op->input_tensors[i]->shape.dim[0], op->input_tensors[i + 1]->shape.dim[0]);
        }
    }
}

template <typename IT> std::string dump_shape(IT first, IT stop)
{
    // at least 1D
    CHECK(first != stop);
    std::stringstream ss;
    ss << "[";
    ss << *first;
    ++first;
    while (first != stop) {
        ss << ", " << *first;
        ++first;
    }
    ss << "]";
    return ss.str();
}

static uint8_t nvinfer_dtype_to_art_dtype(nvinfer1::DataType nv_dtype)
{
    using nvinfer1::DataType;
    switch (nv_dtype) {
    case DataType::kFLOAT:
        return dtFLOAT32;
    case DataType::kHALF:
        return dtFLOAT16;
    case DataType::kINT32:
        return dtINT32;
    case DataType::kINT8:
        return dtINT8;
    case DataType::kBOOL:
        return dtBOOL;

    default:
        LOG_error("unhandled nvinfer dtype `%d` encountered\n", static_cast<int>(nv_dtype));
    }
}

bool op_infer_shape_tensorrt_net(op_t *op)
{
    op_net_t *net = (op_net_t *)op;

    if (net->is_explicit_batch) {
        auto profile_idx = net->nv_net->context_->getOptimizationProfile();
        // bind runtime input dimensions
        for (int i = 0; i < op->input_size; ++i) {
            // first get the runtime shape
            auto tensor = op->input_tensors[i];
            auto &shape = op->input_tensors[i]->shape;
            nvinfer1::Dims dims;
            dims.nbDims = shape.dim_size;
            for (int j = 0; j < shape.dim_size; ++j) {
                dims.d[j] = shape.dim[j];
            }

            int idx = net->nv_net->engine_->getBindingIndex(net->inputs[i].c_str());

            if (!net->nv_net->context_->setBindingDimensions(idx, dims)) {
                // get the min/max shapes of current tensor.
                auto min_dims = net->nv_net->engine_->getProfileDimensions(
                    idx, profile_idx, nvinfer1::OptProfileSelector::kMIN);
                auto max_dims = net->nv_net->engine_->getProfileDimensions(
                    idx, profile_idx, nvinfer1::OptProfileSelector::kMAX);
                auto min_shape = dump_shape(min_dims.d, min_dims.d + min_dims.nbDims);
                auto max_shape = dump_shape(max_dims.d, max_dims.d + max_dims.nbDims);
                auto current_shape = dump_shape(shape.dim, shape.dim + shape.dim_size);
                LOG_warn(
                    "The shape range of tensorrt model's input `%s` is %s (min) to %s (max), "
                    "but current given shape is %s\n",
                    net->inputs[i].c_str(), min_shape.c_str(), max_shape.c_str(),
                    current_shape.c_str());
                return false;
            }
        }
    }

    int i;
    if (0 == op->output_size) {
        op->output_size = net->outputs.size();
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        for (i = 0; i < op->output_size; ++i) {
            op->output_tensors[i].mem = mem_new(op->input_tensors[0]->mem->tp);
            tensor_set_name(&op->output_tensors[i], net->outputs[i].c_str());
        }
    }

    // the offset when map from nvinfer axis to art axis.
    size_t axis_offset = 0;
    if (net->is_explicit_batch) {
        // infer output shape in explicit batch dimension mode.
        axis_offset = 0;
    } else {
        // when not in explicit batch mode, there is an offset=1.
        axis_offset = 1;
    }
    for (i = 0; i < op->output_size; ++i) {
        int idx = net->nv_net->engine_->getBindingIndex(net->outputs[i].c_str());
        nvinfer1::Dims dim = net->nv_net->context_->getBindingDimensions(idx);
        op->output_tensors[i].shape.dim_size = dim.nbDims + axis_offset;
        op->output_tensors[i].shape.batch_axis = 0;
        op->output_tensors[i].shape.channel_axis = 1;

        for (int j = 0; j < dim.nbDims; ++j) {
            op->output_tensors[i].shape.dim[j + axis_offset] = dim.d[j];
        }
        // set dtype
        auto nv_dtype = net->nv_net->engine_->getBindingDataType(idx);
        op->output_tensors[i].dtype = nvinfer_dtype_to_art_dtype(nv_dtype);
    }
    return true;
}

static void op_tensorrt_net_run(op_t *op)
{
    op_net_t *net = (op_net_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        /* sync gpu data */
        mem_data(op->input_tensors[i]->mem);
    }
    for (i = 0; i < op->output_size; ++i) {
        mem_data(op->output_tensors[i].mem);
    }
    // compose I/O buffers in at run, to support replacing input buffer at each run.
    int sz = op->input_size + op->output_size;
    if (net->gpu_buf == nullptr)
        net->gpu_buf = new void *[sz];
    void **itensors_discripter = net->gpu_buf;
    for (size_t i = 0; i < op->input_size; ++i) {
        int idx = net->nv_net->engine_->getBindingIndex(net->inputs[i].c_str());
        itensors_discripter[idx] = mem_data(op->input_tensors[i]->mem);
    }
    for (size_t i = 0; i < op->output_size; ++i) {
        int idx = net->nv_net->engine_->getBindingIndex(net->outputs[i].c_str());
        itensors_discripter[idx] = mem_data(op->output_tensors[i].mem);
    }

    if (net->is_explicit_batch) {
        net->nv_net->context_->executeV2(net->gpu_buf);
    } else {
        int batchSize = op->input_tensors[0]->shape.dim[0];
        net->nv_net->context_->execute(batchSize, net->gpu_buf); //, enqueue(batchSize, buf,
        // TENSORRT_WORKSPACE_STREAM(op->workspace), nullptr);
    }
}

void op_tensorrt_net_tp_prepare(op_t *op)
{
    op_net_t *net = (op_net_t *)op;

    for (size_t i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (size_t i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    op->run_func = op_tensorrt_net_run;
}

void op_tensorrt_net_tp_destroy(op_t *op)
{
    op_net_t *net = (op_net_t *)op;

    if (net->nv_net != nullptr) {
        if (net->nv_net->context_ != nullptr)
            net->nv_net->context_->destroy();

        if (net->nv_net->engine_ != nullptr)
            net->nv_net->engine_->destroy();

        if (net->nv_net->runtime_ != nullptr)
            net->nv_net->runtime_->destroy();

        delete net->nv_net;
    }
    if (net->gpu_buf != nullptr)
        delete[] net->gpu_buf;
    // if (net->netmodel != nullptr)
    //     delete net->netmodel;    //
}
void op_tensorrt_net_tp_dealloc(op_t *op)
{
    if (NULL != op)
        delete (op_net_t *)op;
}
