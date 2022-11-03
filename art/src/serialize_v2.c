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

#include <stdint.h>
#include <string.h>

#include "art/log.h"
#include "art/serialize.h"

#include "./parade_impl.h"
#include "./serialize_impl.h"

/*  serialize v2 format
 *  (head)
 *  <tensor init block>
 *      [int32] tensor count
 *      <input tensors>
 *          [int32] input count
 *          <<repeat `input count` times>>
 *              [uint32] dtype
 *              [string] name
 *              <shape>
 *                  [int32] dim_size
 *                  <<repeat `dim_size` times>>
 *                      [int32] * dim_size
 *                      [int32] batch axis
 *                      [int32] channel axis
 *              [int32] if preprocess param configured
 *              <<if preprocess param configured>>
 *                  <transform_param>
 *                      [uint32_t] operators
 *                      [int32] frame_type
 *                      [pixel_t] mean
 *                      [pixel_t] std
 *                      [pixel_t] padding
 *
 *      <weight tensors>
 *          [int32] weight count
 *          <<repeat `weight count` times>>
 *              [uint32] dtype
 *              [string] name
 *              <shape>
 *                  [int32] dim_size
 *                  <<repeat `dim_size` times>>
 *                      [int32] * dim_size
 *                      [int32] batch axis
 *                      [int32] channel axis
 *              [`shape_count` * sizeof(dtype)] data
 *
 *  <ops>
 *      [int32] op count
 *      <<repeat 'op count' times>>
 *          [int32] op input count
 *          [int32] op output count
 *          [uint64] op type
 *          <<repeat `op input count` times>>
 *              [int32] input tensor index (refer to `tensor count`)
 *          [int32] setting count
 *          <op setting>
 *              <<repeat `setting count` times>>
 *                  [int32] if repeated
 *                  <<if not `if repeated`>>
 *                      [int32] item
 *                      [int32] dtype
 *                      <<if dtype != string>>
 *                          [sizeof(dtype)] value
 *                      <<else>>
 *                          <string>
 *                  <<else>>
 *                      [int32] item
 *                      [int32] dtype
 *                      [int32] count
 *                      <<if dtype != string>>
 *                          [sizeof(dtype) * `count`] data
 *                      <<else>>
 *                          <<repeate `count` times>>
 *                              <string>
 *
 *  <output tensors>
 *      [int32] output count
 *      <<repeat `output count`>>
 *          [int32] index
 *          [string] name
 *
 *  <tensor share>
 *      [int32] share count
 *      <<repeat `share count`>>
 *          [int32] index
 *          [int32] index
 */

typedef struct {
    size_t tensor_count;
    size_t tensor_cur;
    tensor_t **tensors;
} v1_ser;

static workspace_t *_choose_ws(uint64_t op_tp_code, workspace_t *const *workspaces)
{
    workspace_t *const *ws = workspaces;
    if (NULL == ws)
        return NULL;
    while (*ws) {
        if (workspace_support_op(*ws, op_tp_code, true))
            return *ws;
        ws++;
    }
    ws = workspaces;
    while (*ws) {
        if (workspace_support_op(*ws, op_tp_code, false))
            return *ws;
        ws++;
    }
    return NULL;
}

static void _deserialize_op_setting(op_t *op, buffer_t *buf)
{
    bool if_repeated = (bool)deserialize_read_int32(buf);
    uvalue_t u;
    if (false == if_repeated) {
        uint32_t item = deserialize_read_int32(buf);
        uint32_t dtype = deserialize_read_int32(buf);
        switch (dtype) {
        case dtUINT8:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.u8));
            break;
        case dtINT8:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.i8));
            break;
        case dtUINT16:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.u16));
            break;
        case dtINT16:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.i16));
            break;
        case dtUINT32:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.u32));
            break;
        case dtINT32:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.i32));
            break;
        case dtBOOL:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.b));
            break;
        case dtFLOAT16:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.u16));
            break;
        case dtFLOAT32:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.f32));
            break;
        case dtFLOAT64:
            nart_buffer_read(buf, datatype_sizeof(dtype), &u);
            CHECK(op_setting_single_set(op, item, dtype, u.f64));
            break;
        case dtSTR: {
            char *str = deserialize_read_string(buf);
            CHECK(op_setting_single_set(op, item, dtype, str));
            free(str);
            break;
        }
        default:
            CHECK(false);
        }
    } else {
        uint32_t item = deserialize_read_int32(buf);
        uint32_t dtype = deserialize_read_int32(buf);
        size_t count = deserialize_read_int32(buf);
        void *m = malloc(count * datatype_sizeof(dtype));
        if (dtSTR != dtype) {
            nart_buffer_read(buf, datatype_sizeof(dtype) * count, m);
        } else {
            size_t i;
            for (i = 0; i < count; ++i) {
                ((char **)m)[i] = deserialize_read_string(buf);
            }
        }
        CHECK(op_setting_array_set(op, item, dtype, count, m));
        if (dtSTR == dtype) {
            size_t i;
            for (i = 0; i < count; ++i) {
                free(((char **)m)[i]);
            }
        }
        free(m);
    }
}

static void
deserialize_input_tensors(parade_t *parade, v1_ser *ser, buffer_t *buf, const mem_tp *mem_tp)
{
    int32_t input_count = deserialize_read_int32(buf);
    parade->input_tensors = (tensor_t **)malloc(sizeof(tensor_t *) * input_count);
    CHECK(NULL != parade->input_tensors);
    int32_t i;
    for (i = 0; i < input_count; ++i) {
        uint32_t dtype = deserialize_read_int32(buf);
        char *name = deserialize_read_string(buf);
        shape_t shape = deserialize_read_shape(buf);
        tensor_t *tensor = tensor_new(mem_tp, dtype);
        tensor->shape = shape;
        tensor->name = name;

        bool if_transform = (bool)deserialize_read_int32(buf);
        if (if_transform) {
            transform_tensor_t *trans_tensor
                = (transform_tensor_t *)malloc(sizeof(transform_tensor_t));
            memset((void *)trans_tensor, 0, sizeof(transform_tensor_t));
            tensor_share(tensor, (tensor_t *)trans_tensor);
            tensor_set_name(&trans_tensor->tensor, name);
            trans_tensor->tensor.dtype = dtype;
            trans_tensor->tensor.shape = shape;
            trans_tensor->tensor.with_transform = 1;

            uint32_t oprs = deserialize_read_int32(buf);
            int32_t frame_type = deserialize_read_int32(buf);
            pixel_t mean = deserialize_read_pixel(buf);
            pixel_t std = deserialize_read_pixel(buf);
            pixel_t padding = deserialize_read_pixel(buf);

            trans_tensor->param.operators = oprs;
            trans_tensor->param.means = mean;
            trans_tensor->param.stds = std;
            trans_tensor->param.paddings = padding;

            tensor_delete(tensor);
            tensor = (tensor_t *)trans_tensor;
        }

        ser->tensors[ser->tensor_cur++] = (tensor_t *)tensor;
        parade->input_tensors[parade->input_tensor_count++] = (tensor_t *)tensor;
    }
}

static void
deserialize_weight_tensors(parade_t *parade, v1_ser *ser, buffer_t *buf, const mem_tp *mem_tp)
{
    int32_t weight_count = deserialize_read_int32(buf);
    if (weight_count > 0) {
        parade->weight_tensors = (tensor_t **)malloc(sizeof(tensor_t *) * weight_count);
    }
    int32_t i;
    for (i = 0; i < weight_count; ++i) {
        uint32_t dtype = deserialize_read_int32(buf);
        char *name = deserialize_read_string(buf);
        shape_t shape = deserialize_read_shape(buf);
        tensor_t *tensor = tensor_new(mem_tp, dtype);
        tensor->shape = shape;
        tensor->name = name;

        tensor_alloc(tensor);
        void *cpu_data = mem_cpu_data(tensor->mem);
        nart_buffer_read(buf, shape_count(&tensor->shape) * datatype_sizeof(dtype), cpu_data);

        ser->tensors[ser->tensor_cur++] = tensor;
        parade->weight_tensors[parade->weight_tensor_count++] = tensor;
    }
}

static void deserialize_tensor_init_block(
    parade_t *parade, v1_ser *ser, buffer_t *buf, const deserialize_param_t *param)
{
    int32_t tensor_count = deserialize_read_int32(buf);
    ser->tensors = (tensor_t **)malloc(sizeof(tensor_t *) * tensor_count);
    ser->tensor_count = tensor_count;
    deserialize_input_tensors(parade, ser, buf, param->input_mem_tp);
    deserialize_weight_tensors(parade, ser, buf, param->input_mem_tp);
}

static void
deserialize_ops(parade_t *parade, v1_ser *ser, buffer_t *buf, const deserialize_param_t *param)
{
    int32_t op_count = deserialize_read_int32(buf);
    parade->ops = (op_t **)malloc(sizeof(op_t *) * op_count);
    int i;
    for (i = 0; i < op_count; ++i) {
        int32_t op_input_count = deserialize_read_int32(buf);
        int32_t op_output_count = deserialize_read_int32(buf);
        uint64_t op_tp = deserialize_read_int64(buf);
        int j;
        tensor_t **arr_in = (tensor_t **)malloc(sizeof(tensor_t *) * op_input_count);
        for (j = 0; j < op_input_count; ++j) {
            int idx = deserialize_read_int32(buf);
            arr_in[j] = ser->tensors[idx];
        }
        workspace_t *ws = _choose_ws(op_tp, param->workspaces);
        if (ws == NULL) {
            LOG_warn("cannot find the workspace for op, opcode = %lx\n", op_tp);
        }
        if (0 == (op_tp & 0x80000000)) {
            op_tp &= 0xffffffff;
        }
        op_t *op = workspace_new_op(ws, op_tp, op_input_count, arr_in);
        /* settings */
        int setting_count = deserialize_read_int32(buf);
        while (setting_count--) {
            _deserialize_op_setting(op, buf);
        }
        CHECK_EQ(op_config(op), true);
        if (op_output_count != op->output_size) {
            LOG_error(
                "number of op output differs from recorded, expecting %d, got %zu\n",
                op_output_count, op->output_size);
        }
        for (j = 0; j < op_output_count; ++j) {
            ser->tensors[ser->tensor_cur++] = &op->output_tensors[j];
        }
        free(arr_in);

        parade->ops[parade->op_count++] = op;
    }
}

static void deserialize_output_tensors(parade_t *parade, v1_ser *ser, buffer_t *buf)
{
    int output_count = deserialize_read_int32(buf);
    parade->output_tensors = (tensor_t **)malloc(sizeof(tensor_t *) * output_count);
    int i;
    for (i = 0; i < output_count; ++i) {
        int32_t idx = deserialize_read_int32(buf);
        char *name = deserialize_read_string(buf);
        tensor_t *tensor = ser->tensors[idx];
        parade->output_tensors[parade->output_tensor_count++] = tensor;
        tensor_set_name(tensor, name);
        if (NULL != name)
            free(name);
    }
}

static void deserialize_tensor_share(v1_ser *ser, buffer_t *buf)
{
    int share_count = deserialize_read_int32(buf);
    int i;
    for (i = 0; i < share_count; ++i) {
        int idx0 = deserialize_read_int32(buf);
        int idx1 = deserialize_read_int32(buf);
        if (NULL != ser->tensors[idx1]->mem)
            mem_delete(ser->tensors[idx1]->mem);
        ser->tensors[idx1]->mem = mem_dup(ser->tensors[idx0]->mem);
    }
}

parade_t *deserialize_parade_v2(buffer_t *buf, const deserialize_param_t *param)
{
    parade_t *parade = parade_new();
    v1_ser ser;
    memset(&ser, 0, sizeof(ser));
    /* tensor init block */
    deserialize_tensor_init_block(parade, &ser, buf, param);
    /* ops */
    deserialize_ops(parade, &ser, buf, param);
    /* output tensors */
    deserialize_output_tensors(parade, &ser, buf);
    /* tensor share */
    deserialize_tensor_share(&ser, buf);
    free(ser.tensors);
    return parade;
}
