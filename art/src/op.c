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

#include <math.h>
#include <stdarg.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

void op_init_default(
    op_t *op, op_tp_entry_t *entry, workspace_t *workspace, uint16_t input_size,
    tensor_t *const *input_tensor)
{
    op->entry = entry;
    op->workspace = workspace;
    op->input_size = input_size;
    op->input_tensors = (tensor_t **)malloc(sizeof(tensor_t *) * input_size);
    memcpy(op->input_tensors, input_tensor, sizeof(tensor_t *) * input_size);
    op->setting = setting_new();
}

void op_delete(op_t *op)
{
    op->entry->destroy_func(op);
    op_destroy_default(op);
    op->entry->dealloc_func(op);
}

bool op_config(struct op_t *op)
{
    if (NULL != op) {
        op->entry->config_func(op);
        return op_infer_output(op);
    }
    return false;
}

void op_destroy_default(op_t *op)
{
    int i;
    if (op->input_tensors) {
        free(op->input_tensors);
    }
    if (op->output_size > 0) {
        for (i = 0; i < op->output_size; ++i) {
            tensor_free(&op->output_tensors[i]);
        }
        free(op->output_tensors);
    }
    if (NULL != op->setting) {
        setting_delete(op->setting);
        op->setting = NULL;
    }
}

void op_prepare(op_t *op)
{
    setting_shrink(&op->setting);
    op->entry->prepare_func(op);
}

bool op_setting_if_set(const op_t *op, uint32_t item)
{
    if (NULL == op)
        return false;
    if (NULL != setting_search(op->setting, item))
        return true;
    return false;
}

bool op_setting_single_get(const op_t *op, uint32_t item, uint32_t dtype, void *out)
{
    if (NULL == op)
        return false;
    setting_entry_t *setting = setting_search(op->setting, item);
    uvalue_t v;
    if (NULL != setting) {
        if (item != setting->item || dtype != setting->dtype
            || ENUM_SETTING_VALUE_SINGLE != setting->tp) {
            return false;
        }
        v = setting->v.single.value;
    } else { /* NULL == setting */
        const setting_constraint_t *constraint;
        for (constraint = op->entry->tp->constraints; constraint->item != SETTING_END;
             ++constraint) {
            if (item == constraint->item && dtype == constraint->dtype
                && constraint->ctp == ENUM_SETTING_CONSTRAINT_OPTIONAL) {
                v = constraint->constraint.optional.default_value;
                break;
            }
        }
        if (constraint->item == SETTING_END)
            return false;
    }
    switch (dtype) {
    case dtBOOL:
        *(bool *)out = v.b;
        break;
    case dtINT8:
    case dtUINT8:
        *(uint8_t *)out = v.u8;
        break;
    case dtINT16:
    case dtUINT16:
    case dtFLOAT16:
        *(uint16_t *)out = v.u16;
        break;
    case dtINT32:
    case dtUINT32:
    case dtFLOAT32:
        *(uint32_t *)out = v.u32;
        break;
    case dtINT64:
    case dtUINT64:
    case dtFLOAT64:
        *(uint64_t *)out = v.u64;
        break;
    case dtSTR:
    case dtPTR:
    case dtCPTR:
        *(void **)out = v.ptr;
        break;
    default:
        break;
    }
    return true;
}

static const setting_constraint_t *
setting_constraint_search(const setting_constraint_t *constraints, uint32_t item)
{
    if (NULL == constraints)
        return NULL;
    while (constraints->item != SETTING_END && constraints->item != item)
        constraints++;
    if (constraints->item == SETTING_END)
        return NULL;
    else
        return constraints;
}

bool op_setting_single_set(op_t *op, uint32_t item, uint32_t dtype, ...)
{
    va_list valist;
    if (NULL == op)
        return false;
    const setting_constraint_t *constraint
        = setting_constraint_search(op->entry->tp->constraints, item);
    if (NULL != constraint) {
        if (constraint->dtype != dtype || constraint->ctp == ENUM_SETTING_CONSTRAINT_REPEATED)
            return false;
    }
    va_start(valist, dtype);
    uvalue_t v;
    switch (dtype) {
    case dtBOOL:
        // v.b = *(const bool*)in;
        v.b = (bool)va_arg(valist, uint32_t);
        break;
    case dtINT8:
    case dtUINT8:
        // v.u8 = *(const uint8_t*)in;
        v.u8 = (uint8_t)va_arg(valist, uint32_t);
        break;
    case dtINT16:
    case dtUINT16:
    case dtFLOAT16:
        // v.u16 = *(const uint16_t*)in;
        v.u16 = (uint16_t)va_arg(valist, uint32_t);
        break;
    case dtINT32:
    case dtUINT32:
        // v.u32 = *(const uint16_t*)in;
        v.u32 = va_arg(valist, uint32_t);
        break;
    case dtFLOAT32:
        v.f32 = (float)va_arg(valist, double);
        break;
    case dtINT64:
    case dtUINT64:
    case dtFLOAT64:
        // v.u64 = *(const uint16_t*)in;
        v.u64 = va_arg(valist, uint64_t);
        break;
    case dtSTR:
    case dtPTR:
    case dtCPTR:
        // v.cptr = in;
        v.cptr = va_arg(valist, void *);
        break;
    default:
        break;
    }
    va_end(valist);
    setting_set_single(&op->setting, item, dtype, v);
    return true;
}

bool op_setting_array_set(op_t *op, uint32_t item, uint32_t dtype, size_t len, const void *in)
{
    if (NULL == op)
        return false;
    const setting_constraint_t *constraint
        = setting_constraint_search(op->entry->tp->constraints, item);
    if (NULL != constraint) {
        if (constraint->dtype != dtype || constraint->ctp != ENUM_SETTING_CONSTRAINT_REPEATED)
            return false;
    }
    void *res = setting_alloc_repeated(&op->setting, item, dtype, len);
    if (NULL == res)
        return false;
    if (dtype != dtSTR) {
        memcpy(res, in, datatype_sizeof(dtype) * len);
    } else {
        size_t i;
        for (i = 0; i < len; ++i) {
            char *str = (char *)malloc(sizeof(char) * strlen(((const char **)in)[i]) + 1);
            CHECK_ACTION(NULL != str) return false;
            ((char **)res)[i] = str;
            strcpy(str, ((const char **)in)[i]);
        }
    }
    return true;
}

bool op_setting_array_append(op_t *op, uint32_t item, uint32_t dtype, size_t len, const void *in);

bool op_setting_array_get(const op_t *op, uint32_t item, uint32_t dtype, size_t *len, void *out)
{
    if (NULL == op)
        return false;
    size_t length = 0;
    void *o = NULL;
    /* search settings first */
    const setting_entry_t *entry = op->setting->entries;
    size_t i;
    for (i = 0; i < op->setting->len; ++entry, ++i) {
        if (item == entry->item) {
            if (dtype != entry->dtype || entry->tp != ENUM_SETTING_VALUE_REPEATED)
                return false;
            length = entry->v.repeated.len;
            o = entry->v.repeated.values;
            goto ret_v;
        }
    }
    /* no setting found, search constraints */
    const setting_constraint_t *constraint
        = setting_constraint_search(op->entry->tp->constraints, item);
    if (NULL == constraint)
        return false;
    if (constraint->ctp != ENUM_SETTING_CONSTRAINT_REPEATED)
        return false;
ret_v:
    if (NULL != len)
        *len = length;
    if (NULL != out)
        *(void **)out = o;
    return true;
}

static bool op_infer_shape_unary_operator(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
        op->output_tensors[0].shape.channel_axis = op->input_tensors[0]->shape.channel_axis;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_binary_operator(op_t *op)
{
    CHECK_EQ(2, op->input_size);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
        op->output_tensors[0].shape.channel_axis = op->input_tensors[0]->shape.channel_axis;
    }
    size_t dim1 = op->input_tensors[0]->shape.dim_size;
    size_t dim2 = op->input_tensors[1]->shape.dim_size;
    size_t dim = dim1 > dim2 ? dim1 : dim2;
    shape_t shape_0 = op->input_tensors[0]->shape;
    shape_t shape_1 = op->input_tensors[1]->shape;

    shape_0.dim_size = dim;
    shape_1.dim_size = dim;

    int i;
    for (i = 0; i < dim1; ++i) {
        shape_0.dim[dim - 1 - i] = shape_0.dim[dim1 - 1 - i];
    }
    for (; i < dim; ++i) {
        shape_0.dim[dim - 1 - i] = 1;
    }
    for (i = 0; i < dim2; ++i) {
        shape_1.dim[dim - 1 - i] = shape_1.dim[dim2 - 1 - i];
    }
    for (; i < dim; ++i) {
        shape_1.dim[dim - 1 - i] = 1;
    }

    op->output_tensors[0].shape = shape_0;
    for (i = 0; i < dim; ++i) {
        if (shape_0.dim[i] == shape_1.dim[i])
            op->output_tensors[0].shape.dim[i] = shape_0.dim[i];
        else if (shape_0.dim[i] == 1)
            op->output_tensors[0].shape.dim[i] = shape_1.dim[i];
        else if (shape_1.dim[i] == 1)
            op->output_tensors[0].shape.dim[i] = shape_0.dim[i];
        else
            LOG(error, "cannot broadcast %d dim: %d vs %d\n", i, shape_0.dim[i], shape_1.dim[i]);
    }

    return true;
}

static bool op_infer_shape_quant_dequant(op_t *op)
{
    CHECK_EQ(2, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[1]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[1]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[1]->shape;
    return true;
}

static bool op_infer_shape_bn(op_t *op)
{
    CHECK_LE(3, op->input_size);
    CHECK_GE(5, op->input_size);
    CHECK_NE(4, op->input_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_batchnorm(op_t *op)
{
    CHECK_LE(1, op->input_size);
    // CHECK_GE(3, op->input_size);
    CHECK_NE(2, op->input_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_conv_nd(op_t *op)
{
    CHECK_LE(2, op->input_size);
    CHECK_GE(3, op->input_size);

    int n = op->input_tensors[0]->shape.dim_size - 2;
    uint32_t group = 0;
    uint32_t num_output = 0;
    uint32_t *kernel = NULL;
    uint32_t *pad = NULL;
    uint32_t *stride = NULL;
    uint32_t *hole = NULL;
    size_t i;

    CHECK(op_setting_single_get(op, SETTING_CONV_ND_GROUP, dtUINT32, &group));
    CHECK(op_setting_single_get(op, SETTING_CONV_ND_NUM_OUTPUT, dtUINT32, &num_output));
    CHECK(op_setting_array_get(op, SETTING_CONV_ND_PAD, dtUINT32, NULL, &pad));
    CHECK(op_setting_array_get(op, SETTING_CONV_ND_KERNEL, dtUINT32, NULL, &kernel));
    CHECK(op_setting_array_get(op, SETTING_CONV_ND_STRIDE, dtUINT32, NULL, &stride));
    CHECK(op_setting_array_get(op, SETTING_CONV_ND_HOLE, dtUINT32, NULL, &hole));

    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK(
        1 == op->input_tensors[0]->shape.channel_axis
        || n + 1 == op->input_tensors[0]->shape.channel_axis);
    CHECK_EQ(n + 2, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(n + 2, op->input_tensors[1]->shape.dim_size);

    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int spatial_dims[MAX_DIM];
    if (1 == c_dim) {
        /* input:  n c (...) */
        /* kernel: n c (...) */
        for (i = 0; i < n; i++) {
            spatial_dims[i] = i + 2;
        }
        CHECK_EQ(0, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(1, op->input_tensors[1]->shape.channel_axis);
    } else {
        /* input:  n (...) c */
        /* kernel: (...) c n */
        for (i = 0; i < n; i++) {
            spatial_dims[i] = i + 1;
        }
        CHECK_EQ(n + 2, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(n + 1, op->input_tensors[1]->shape.channel_axis);
    }
    /* check group */
    CHECK_EQ(0, op->input_tensors[0]->shape.dim[c_dim] % group);
    CHECK_EQ(0, num_output % group);
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim] / group,
        op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.channel_axis]);

    /* check weight */
    CHECK_EQ(num_output, op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis]);
    if (op->input_size >= 3) {
        CHECK_EQ(1, op->input_tensors[2]->shape.dim_size);
        CHECK_EQ(num_output, op->input_tensors[2]->shape.dim[0]);
    }

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[output_shape.channel_axis] = num_output;

    uint32_t kernel_eff;
    for (i = 0; i < n; i++) {
        kernel_eff = kernel[i] + (kernel[i] - 1) * (hole[i] - 1);
        output_shape.dim[spatial_dims[i]]
            = (op->input_tensors[0]->shape.dim[spatial_dims[i]] + pad[i] * 2 - kernel_eff)
                / stride[i]
            + 1;
    }

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;

    return true;
}

static bool op_infer_shape_conv_2d(op_t *op)
{
    CHECK_LE(2, op->input_size);
    CHECK_GE(3, op->input_size);
    // for (i = 1; i < op->input_size; ++i) {
    //     CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[i]->dtype);
    // }

    uint32_t group = 0;
    uint32_t num_output = 0;
    uint32_t kernel_h = 0, kernel_w = 0;
    uint32_t pad_h = 0, pad_w = 0;
    uint32_t stride_h = 0, stride_w = 0;
    uint32_t hole_h = 0, hole_w = 0;
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &group));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &num_output));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &pad_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &kernel_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &stride_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &stride_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_HOLE_H, dtUINT32, &hole_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_HOLE_W, dtUINT32, &hole_w));

    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK(
        1 == op->input_tensors[0]->shape.channel_axis
        || 3 == op->input_tensors[0]->shape.channel_axis);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(4, op->input_tensors[1]->shape.dim_size);

    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim;
    int w_dim;
    if (1 == c_dim) {
        /* input:  n c h w */
        /* kernel: n c h w */
        h_dim = 2;
        w_dim = 3;
        CHECK_EQ(0, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(1, op->input_tensors[1]->shape.channel_axis);
    } else {
        /* input:  n h w c */
        /* kernel: h w c n */
        h_dim = 1;
        w_dim = 2;
        CHECK_EQ(3, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(2, op->input_tensors[1]->shape.channel_axis);
    }

    /* check group */
    CHECK_EQ(0, op->input_tensors[0]->shape.dim[c_dim] % group);
    CHECK_EQ(0, num_output % group);
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim] / group,
        op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.channel_axis]);

    /* check weight */
    CHECK_EQ(num_output, op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis]);
    if (op->input_size >= 3) {
        CHECK_EQ(1, op->input_tensors[2]->shape.dim_size);
        CHECK_EQ(num_output, op->input_tensors[2]->shape.dim[0]);
    }

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[output_shape.channel_axis] = num_output;

    uint32_t kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
    uint32_t kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);

    output_shape.dim[h_dim]
        = (op->input_tensors[0]->shape.dim[h_dim] + pad_h * 2 - kernel_h_eff) / stride_h + 1;
    output_shape.dim[w_dim]
        = (op->input_tensors[0]->shape.dim[w_dim] + pad_w * 2 - kernel_w_eff) / stride_w + 1;

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_deform_conv_2d(op_t *op)
{
    CHECK_EQ(3, op->input_size);
    // for (i = 1; i < op->input_size; ++i) {
    //     CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[i]->dtype);
    // }

    uint32_t group = 0;
    uint32_t deform_group = 0;
    uint32_t num_output = 0;
    uint32_t hole_h = 0, hole_w = 0;
    uint32_t kernel_h = 0, kernel_w = 0;
    uint32_t pad_h = 0, pad_w = 0;
    uint32_t stride_h = 0, stride_w = 0;
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &group));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &num_output));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &pad_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &kernel_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &stride_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &stride_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_DEFORM_GROUP, dtUINT32, &deform_group));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_HOLE_H, dtUINT32, &hole_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_HOLE_W, dtUINT32, &hole_w));

    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK(
        1 == op->input_tensors[0]->shape.channel_axis
        || 3 == op->input_tensors[0]->shape.channel_axis);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(4, op->input_tensors[1]->shape.dim_size);

    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim;
    int w_dim;
    if (1 == c_dim) {
        /* input:  n c h w */
        /* kernel: n c h w */
        h_dim = 2;
        w_dim = 3;
        CHECK_EQ(0, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(1, op->input_tensors[1]->shape.channel_axis);
    } else {
        /* input:  n h w c */
        /* kernel: h w c n */
        h_dim = 1;
        w_dim = 2;
        CHECK_EQ(3, op->input_tensors[1]->shape.batch_axis);
        CHECK_EQ(2, op->input_tensors[1]->shape.channel_axis);
    }

    /* check group */
    CHECK_EQ(0, op->input_tensors[0]->shape.dim[c_dim] % group);
    CHECK_EQ(0, num_output % group);
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim] / group,
        op->input_tensors[2]->shape.dim[op->input_tensors[1]->shape.channel_axis]);

    /* check weight */
    CHECK_EQ(num_output, op->input_tensors[2]->shape.dim[op->input_tensors[1]->shape.batch_axis]);

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[output_shape.channel_axis] = num_output;

    output_shape.dim[h_dim]
        = (op->input_tensors[0]->shape.dim[h_dim] + pad_h * 2 - hole_h * (kernel_h - 1) - 1)
            / stride_h
        + 1;
    output_shape.dim[w_dim]
        = (op->input_tensors[0]->shape.dim[w_dim] + pad_w * 2 - hole_w * (kernel_w - 1) - 1)
            / stride_w
        + 1;

    /* check offset */
    CHECK_EQ(4, op->input_tensors[1]->shape.dim_size);
    CHECK_EQ(
        op->input_tensors[1]->shape.dim[c_dim],
        2 * kernel_h * kernel_w * op->input_tensors[0]->shape.dim[c_dim]);
    /* redundant check */
    // CHECK_EQ(op->input_tensors[1]->shape.dim[h_dim], output_shape.dim[h_dim]);
    // CHECK_EQ(op->input_tensors[1]->shape.dim[w_dim], output_shape.dim[w_dim]);

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_deconv_2d(op_t *op)
{
    CHECK_LE(2, op->input_size);
    CHECK_GE(3, op->input_size);

    uint32_t group = 0;
    uint32_t num_output = 0;
    uint32_t kernel_h = 0, kernel_w = 0;
    uint32_t pad_h = 0, pad_w = 0;
    uint32_t stride_h = 0, stride_w = 0;
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &group));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &num_output));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &pad_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &kernel_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &stride_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &stride_w));

    // int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim = 0;
    int w_dim = 0;
    while (h_dim == op->input_tensors[0]->shape.batch_axis
           || h_dim == op->input_tensors[0]->shape.channel_axis) {
        h_dim++;
    }
    w_dim = h_dim + 1;

    CHECK_EQ(0, op->input_tensors[0]->shape.dim[c_dim] % group);
    CHECK_EQ(0, num_output % group);

    CHECK_GE(1, op->output_size);

    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(4, op->input_tensors[1]->shape.dim_size);
    if (op->input_size >= 3) {
        CHECK_EQ(1, op->input_tensors[2]->shape.dim_size);
    }

    if (op->input_size >= 3) {
        CHECK_EQ(num_output, shape_count(&op->input_tensors[2]->shape));
    }

    // infer the output shape

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[op->input_tensors[0]->shape.channel_axis] = num_output;
    output_shape.dim[h_dim]
        = (op->input_tensors[0]->shape.dim[h_dim] - 1) * stride_h + kernel_h - pad_h * 2;
    output_shape.dim[w_dim]
        = (op->input_tensors[0]->shape.dim[w_dim] - 1) * stride_w + kernel_w - pad_w * 2;

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
    }
    if (op->input_tensors[0]->dtype == dtUINT8 || op->input_tensors[0]->dtype == dtINT8) {
        op->output_tensors[0].dtype = dtINT32;
    } else {
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_correlation(op_t *op)
{
    /* input_tensors[0]: kernel
     * input_tensors[1]: input
     */
    CHECK_EQ(2, op->input_size);
    uint32_t batch_size = 0;
    uint32_t input_h = 0, input_w = 0;
    uint32_t kernel_h = 0, kernel_w = 0;
    uint32_t channel_in = 0, channel_out = 0;
    uint32_t groups;

    CHECK(op_setting_single_get(op, SETTING_CORRELATION_GROUPS, dtUINT32, &groups));
    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK(
        1 == op->input_tensors[0]->shape.channel_axis
        || 3 == op->input_tensors[0]->shape.channel_axis);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(4, op->input_tensors[1]->shape.dim_size);

    int c_dim = op->input_tensors[0]->shape.channel_axis;

    int h_dim, w_dim;
    if (1 == c_dim) {
        /* input:  n c h w */
        /* kernel: n c h w */
        h_dim = 2;
        w_dim = 3;
        CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
        CHECK_EQ(1, op->input_tensors[0]->shape.channel_axis);
    } else {
        /* input:  n h w c */
        /* kernel: h w c n */
        h_dim = 1;
        w_dim = 2;
        CHECK_EQ(3, op->input_tensors[0]->shape.batch_axis);
        CHECK_EQ(2, op->input_tensors[0]->shape.channel_axis);
    }

    batch_size = op->input_tensors[1]->shape.dim[op->input_tensors[0]->shape.batch_axis];
    input_h = op->input_tensors[0]->shape.dim[h_dim];
    input_w = op->input_tensors[0]->shape.dim[w_dim];
    kernel_h = op->input_tensors[1]->shape.dim[h_dim];
    kernel_w = op->input_tensors[1]->shape.dim[w_dim];
    channel_in = op->input_tensors[0]->shape.dim[c_dim];
    channel_out = op->input_tensors[1]->shape.dim[c_dim];

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[h_dim] = input_h - kernel_h + 1;
    output_shape.dim[w_dim] = input_w - kernel_w + 1;
    output_shape.dim[output_shape.batch_axis] = batch_size;

    if (groups == 1) {
        output_shape.dim[c_dim] = channel_out / channel_in;
    } else {
        output_shape.dim[c_dim] = channel_in;
    }

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_default_ip(op_t *op)
{
    int i;
    CHECK_LE(2, op->input_size);
    CHECK_GE(3, op->input_size);
    // for (i = 1; i < op->input_size; ++i) {
    //     CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[i]->dtype);
    // }
    CHECK_GE(1, op->output_size);

    if (op->input_size >= 3) {
        CHECK_EQ(1, op->input_tensors[2]->shape.dim_size);
    }

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim = 0;
    int w_dim = 0;
    while (h_dim == op->input_tensors[0]->shape.batch_axis
           || h_dim == op->input_tensors[0]->shape.channel_axis) {
        h_dim++;
    }
    w_dim = h_dim + 1;

    // CHECK_EQ(op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.channel_axis],
    // shape_count(&op->input_tensors[0]->shape) / op->input_tensors[0]->shape.dim[n_dim]);
    if (op->input_size >= 3) {
        CHECK_EQ(
            op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis],
            op->input_tensors[2]->shape.dim[op->input_tensors[2]->shape.batch_axis]);
    }
    uint32_t num_output;
    int32_t axis;
    CHECK(op_setting_single_get(op, SETTING_IP_NUM_OUTPUT, dtUINT32, &num_output));
    CHECK(op_setting_single_get(op, SETTING_IP_AXIS, dtINT32, &axis));
    int dim_size = op->input_tensors[0]->shape.dim_size;
    axis = axis < 0 ? axis + dim_size : axis;
    CHECK_GT(dim_size, axis);
    CHECK_LE(0, axis);

    uint32_t dim[MAX_DIM], preceding_axes = 1, succeeding_axes = 1;
    dim[0] = op->input_tensors[0]->shape.dim[n_dim];
    dim[1] = op->input_tensors[0]->shape.dim[c_dim];
    dim[2] = op->input_tensors[0]->shape.dim[h_dim];
    dim[3] = op->input_tensors[0]->shape.dim[w_dim];
    for (i = 0; i < axis; ++i) {
        preceding_axes *= dim[i];
    }
    for (i = axis; i < dim_size; ++i) {
        succeeding_axes *= dim[i];
    }
    CHECK_EQ(op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis], num_output);
    CHECK_EQ(shape_count(&op->input_tensors[1]->shape) / num_output, succeeding_axes);

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    shape_t shape = op->input_tensors[0]->shape;
    shape.dim_size = axis + 1;
    shape.batch_axis = n_dim;
    shape.channel_axis = axis;
    shape.dim[axis] = num_output;
    for (i = axis + 1; i < MAX_DIM; ++i) {
        shape.dim[i] = 1;
    }
    op->output_tensors[0].shape = shape;
    return true;
}

static bool op_infer_shape_concat(op_t *op)
{
    int i, j;
    int axis;
    CHECK(op_setting_single_get(op, SETTING_CONCAT_AXIS, dtUINT32, &axis));
    CHECK_LE(2, op->input_size);
    CHECK_GE(1, op->output_size);
    CHECK_LT(axis, op->input_tensors[0]->shape.dim_size);
    for (i = 1; i < op->input_size; ++i) {
        CHECK_EQ(op->input_tensors[0]->shape.dim_size, op->input_tensors[i]->shape.dim_size);
        CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[i]->dtype);
        for (j = 0; j < op->input_tensors[0]->shape.dim_size; ++j) {
            if (j != axis) {
                CHECK_EQ(op->input_tensors[0]->shape.dim[j], op->input_tensors[i]->shape.dim[j]);
            }
        }
    }
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    for (i = 1; i < op->input_size; ++i) {
        op->output_tensors[0].shape.dim[axis] += op->input_tensors[i]->shape.dim[axis];
    }
    return true;
}

static bool op_infer_shape_pool(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK_GE(1, op->output_size);

    // infer the output shape
    shape_t output_shape;
    output_shape.batch_axis = op->input_tensors[0]->shape.batch_axis;
    output_shape.channel_axis = op->input_tensors[0]->shape.channel_axis;
    output_shape.dim_size = 4;
    output_shape.dim[output_shape.batch_axis]
        = op->input_tensors[0]->shape.dim[output_shape.batch_axis];
    output_shape.dim[output_shape.channel_axis]
        = op->input_tensors[0]->shape.dim[output_shape.channel_axis];
    int h_dim = 0;
    while (h_dim == output_shape.batch_axis || h_dim == output_shape.channel_axis) {
        ++h_dim;
    }
    int w_dim = 3;
    while (w_dim == output_shape.batch_axis || w_dim == output_shape.channel_axis) {
        --w_dim;
    }
    bool ceil_mode;
    uint32_t pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w;
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_H, dtUINT32, &kernel_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_W, dtUINT32, &kernel_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_H, dtUINT32, &pad_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_W, dtUINT32, &pad_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_H, dtUINT32, &stride_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_W, dtUINT32, &stride_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_CEIL_MODE, dtBOOL, &ceil_mode));
    if (ceil_mode) {
        output_shape.dim[h_dim] = ceil(
            (op->input_tensors[0]->shape.dim[h_dim] + pad_h * 2.0 - kernel_h) / stride_h + 1);
        output_shape.dim[w_dim] = ceil(
            (op->input_tensors[0]->shape.dim[w_dim] + pad_w * 2.0 - kernel_w) / stride_w + 1);
    } else {
        output_shape.dim[h_dim] = floor(
            (op->input_tensors[0]->shape.dim[h_dim] + pad_h * 2.0 - kernel_h) / stride_h + 1);
        output_shape.dim[w_dim] = floor(
            (op->input_tensors[0]->shape.dim[w_dim] + pad_w * 2.0 - kernel_w) / stride_w + 1);
    }

    // ensure the last pooling starts strictly inside the image
    if (pad_h || pad_w) {
        if ((output_shape.dim[h_dim] - 1) * stride_h
            >= op->input_tensors[0]->shape.dim[h_dim] + pad_h) {
            output_shape.dim[h_dim]--;
        }
        if ((output_shape.dim[w_dim] - 1) * stride_w
            >= op->input_tensors[0]->shape.dim[w_dim] + pad_w) {
            output_shape.dim[w_dim]--;
        }
    }

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_default_interp(op_t *op)
{
    int i;
    CHECK_LE(1, op->input_size);
    CHECK_GE(2, op->input_size);
    for (i = 1; i < op->input_size; ++i) {
        CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[i]->dtype);
    }

    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);

    uint32_t zoom_factor, shrink_factor, height, width, pad_beg, pad_end;
    CHECK(op_setting_single_get(op, SETTING_INTERP_ZOOM_FACTOR, dtUINT32, &zoom_factor));
    CHECK(op_setting_single_get(op, SETTING_INTERP_SHRINK_FACTOR, dtUINT32, &shrink_factor));
    CHECK(op_setting_single_get(op, SETTING_INTERP_HEIGHT, dtUINT32, &height));
    CHECK(op_setting_single_get(op, SETTING_INTERP_WIDTH, dtUINT32, &width));
    CHECK(op_setting_single_get(op, SETTING_INTERP_PAD_BEG, dtUINT32, &pad_beg));
    CHECK(op_setting_single_get(op, SETTING_INTERP_PAD_END, dtUINT32, &pad_end));

    shape_t shape = op->input_tensors[0]->shape;

    int h_dim = 0;
    while (h_dim == shape.batch_axis || h_dim == shape.channel_axis) {
        ++h_dim;
    }
    int w_dim = 3;
    while (w_dim == shape.batch_axis || w_dim == shape.channel_axis) {
        --w_dim;
    }

    int i_h = op->input_tensors[0]->shape.dim[h_dim];
    int i_w = op->input_tensors[0]->shape.dim[w_dim];
    int i_h_eff = i_h - pad_beg - pad_end;
    int i_w_eff = i_w - pad_beg - pad_end;
    int o_h = 0, o_w = 0;

    if (op->input_size > 1) {
        o_h = op->input_tensors[1]->shape.dim[h_dim];
        o_w = op->input_tensors[1]->shape.dim[w_dim];
    } else if (0 != zoom_factor) {
        o_h = i_h_eff + (i_h_eff - 1) * (zoom_factor - 1);
        o_w = i_w_eff + (i_w_eff - 1) * (zoom_factor - 1);
    } else if (0 != shrink_factor) {
        o_h = (i_h_eff - 1) / shrink_factor + 1;
        o_w = (i_w_eff - 1) / shrink_factor + 1;
    } else if (height != 0 && width != 0) {
        o_h = height;
        o_w = width;
    } else {
    }
    CHECK_LT(0, o_h);
    CHECK_LT(0, o_w);
    shape.dim[h_dim] = o_h;
    shape.dim[w_dim] = o_w;

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = shape;
    return true;
}

static bool op_infer_shape_slice(op_t *op)
{
    uint32_t axis;
    CHECK(op_setting_single_get(op, SETTING_SLICE_AXIS, dtUINT32, &axis));
    CHECK_EQ(1, op->input_size);
    uint32_t *var;
    size_t len;
    CHECK(op_setting_array_get(op, SETTING_SLICE_POINT, dtUINT32, &len, &var));
    CHECK_LT(0, len);
    {
        size_t i;
        for (i = 0; i < len - 1; ++i) {
            CHECK_LT(var[i], var[i + 1]);
        }
    }
    if (0 == op->output_size) {
        op->output_size = len + 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        int i;
        for (i = 0; i < op->output_size; ++i) {
            op->output_tensors[i].mem = mem_new(op->input_tensors[0]->mem->tp);
            op->output_tensors[i].dtype = op->input_tensors[0]->dtype;
        }
    }
    int i;
    for (i = 0; i < op->output_size; ++i) {
        op->output_tensors[i].shape = op->input_tensors[0]->shape;
        if (0 == i) {
            op->output_tensors[i].shape.dim[axis] = var[i];
        } else if (i == op->output_size - 1) {
            op->output_tensors[i].shape.dim[axis]
                = op->input_tensors[0]->shape.dim[axis] - var[i - 1];
        } else {
            op->output_tensors[i].shape.dim[axis] = var[i] - var[i - 1];
        }
    }
    return true;
}

static bool op_infer_shape_eltwise(op_t *op)
{
    CHECK_GE(op->input_size, 2);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    CHECK_EQ(op->output_size, 1);
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_scale(op_t *op)
{
    // CHECK_EQ(3, op->input_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_reshape(op_t *op)
{
    int shape[MAX_DIM] = { 0 };
    int32_t axis;

    CHECK(op_setting_single_get(op, SETTING_RESHAPE_AXIS, dtINT32, &axis));
    CHECK_EQ(1, op->input_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_dup(op->input_tensors[0]->mem);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }

    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    int32_t *var;
    size_t num_axes;
    CHECK(op_setting_array_get(op, SETTING_RESHAPE_DIMS, dtINT32, &num_axes, &var));
    op->output_tensors[0].shape.dim_size = num_axes;
    int infer_axis = 0;
    {
        int i;
        axis = axis < 0 ? axis + op->input_tensors[0]->shape.dim_size : axis;
        CHECK_LE(0, axis);
        num_axes
            = (num_axes == 0) ? (size_t)(op->input_tensors[0]->shape.dim_size - axis) : (num_axes);
        for (i = num_axes - 1; i >= 0; i--) {
            shape[axis + i] = var[i];
            if (var[i] == -1)
                infer_axis = axis + i;
            if (var[i] == 0) {
                shape[axis + i] = op->input_tensors[0]->shape.dim[axis + i];
            }
        }
    }
    if (infer_axis != -1) {
        int all = 1;
        int i;
        for (i = 0; i < op->input_tensors[0]->shape.dim_size; ++i) {
            all *= op->input_tensors[0]->shape.dim[i];
        }
        for (i = 0; i < (int)num_axes; ++i) {
            if (i != infer_axis)
                all /= shape[i];
        }
        shape[infer_axis] = all;
    }
    // FIXME:
    //     reshape support nhwc layout
    // if (op->input_tensors[0]->shape.channel_axis > 0) {
    //    if (op->input_tensors[0]->shape.channel_axis == op->input_tensors[0]->shape.dim_size - 1)
    //    {
    //        op->output_tensors[0].shape.channel_axis = op->output_tensors[0].shape.dim_size - 1;
    //    }
    //}
    tensor_reshape(&op->output_tensors[0], op->output_tensors[0].shape.dim_size, shape);
    return true;
}

static bool op_infer_shape_subpixel(op_t *op)
{
    CHECK_EQ(1, op->input_size);

    uint8_t method;
    uint32_t sample;
    CHECK(op_setting_single_get(op, SETTING_SUBPIXEL_METHOD, dtUINT8, &method));
    CHECK(op_setting_single_get(op, SETTING_SUBPIXEL_SAMPLE, dtUINT32, &sample));
    CHECK_GE(1, method);

    shape_t shape = op->input_tensors[0]->shape;
    int h_dim = 0, w_dim = 3;
    while (h_dim == shape.batch_axis || h_dim == shape.channel_axis) {
        ++h_dim;
    }
    while (w_dim == shape.batch_axis || w_dim == shape.channel_axis) {
        --w_dim;
    }

    if (0 == method) {
        CHECK_EQ(0, op->input_tensors[0]->shape.dim[h_dim] % sample);
        CHECK_EQ(0, op->input_tensors[0]->shape.dim[w_dim] % sample);
    } else {
        CHECK_EQ(0, op->input_tensors[0]->shape.dim[shape.channel_axis] % (sample * sample));
    }

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }

    shape_t output_shape = shape;
    if (0 == method) {
        output_shape.dim[output_shape.channel_axis]
            = shape.dim[shape.channel_axis] * (sample * sample);
        output_shape.dim[h_dim] = shape.dim[h_dim] / sample;
        output_shape.dim[w_dim] = shape.dim[w_dim] / sample;
    } else {
        output_shape.dim[output_shape.channel_axis]
            = shape.dim[shape.channel_axis] / (sample * sample);
        output_shape.dim[h_dim] = shape.dim[h_dim] * sample;
        output_shape.dim[w_dim] = shape.dim[w_dim] * sample;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_transpose(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }

    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    int32_t *dims;
    size_t ndim;
    bool flag_exgaxis;
    CHECK(op_setting_array_get(op, SETTING_TRANSPOSE_DIMS, dtINT32, &ndim, &dims));
    CHECK(op_setting_single_get(op, SETTING_TRANSPOSE_EXGAXIS, dtBOOL, &flag_exgaxis));
    CHECK_EQ(op->input_tensors[0]->shape.dim_size, (int)ndim);

    size_t i;
    for (i = 0; i < ndim; ++i) {
        op->output_tensors[0].shape.dim[i] = op->input_tensors[0]->shape.dim[dims[i]];
        if (flag_exgaxis) {
            if (dims[i] == op->input_tensors[0]->shape.batch_axis) {
                shape_set_batch_axis(&op->output_tensors[0].shape, i);
            }
            if (dims[i] == op->input_tensors[0]->shape.channel_axis) {
                shape_set_channel_axis(&op->output_tensors[0].shape, i);
            }
        }
    }
    return true;
}

static bool op_infer_shape_prelu(op_t *op)
{
    CHECK_EQ(2, op->input_size);
    CHECK_GE(1, op->output_size);
    CHECK_LE(2, op->input_tensors[0]->shape.dim_size);
    CHECK_EQ(1, op->input_tensors[1]->shape.dim_size);

    bool share;
    CHECK(op_setting_single_get(op, SETTING_PRELU_SHARE, dtBOOL, &share));
    if (share) {
        CHECK_EQ(1, op->input_tensors[1]->shape.dim[0]);
    } else {
        CHECK_EQ(
            op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.channel_axis],
            op->input_tensors[1]->shape.dim[0]);
    }

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_matmul(op_t *op)
{
    // example:
    //   data:   input[0].shape = (m, k)
    //   weight: input[1].shape = (k, n)
    //   bias:   input[2].shape = (n)    or no bias
    // these are allowed, and then
    //   output[0].shape = (m, n)
    CHECK(2 <= op->input_size && op->input_size <= 3);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    size_t size = op->input_tensors[0]->shape.dim_size;
    size_t size2 = op->input_tensors[1]->shape.dim_size;
    if (size != size2) {
        LOG_error("Matmul requires both inputs to have same number of dimensions\n");
    }
    if (op->input_size == 3) {
        CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[2]->dtype);
        CHECK_EQ(1, op->input_tensors[2]->shape.dim_size);
        CHECK_EQ(op->input_tensors[1]->shape.dim[1], op->input_tensors[2]->shape.dim[0]);
    }
    CHECK(1 >= op->output_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
        // the channel axis of MatMul's output is always the last axis. But because
        // transpose op's exgaxis flag is not enabled, we have to set the channel axis
        // to 1, otherwise the infered channel aixs will be wrong when transpose is involved.
        // op->output_tensors[0].shape.channel_axis = size - 1;
        op->output_tensors[0].shape.channel_axis = 1;
    }
    shape_t *shape0 = &op->input_tensors[0]->shape;
    shape_t *shape1 = &op->input_tensors[1]->shape;
    CHECK_EQ(shape0->dim[shape0->dim_size - 1], shape1->dim[shape1->dim_size - 2]);
    uint32_t m = shape0->dim[shape0->dim_size - 2];
    uint32_t n = shape1->dim[shape1->dim_size - 1];
    shape_t *output_shape = &op->output_tensors[0].shape;
    output_shape->dim_size = size;
    for (size_t axis = 0; axis < size - 2; ++axis) {
        if (shape0->dim[axis] == 1) {
            output_shape->dim[axis] = shape1->dim[axis];
        } else if (shape1->dim[axis] == 1) {
            output_shape->dim[axis] = shape0->dim[axis];
        } else if (shape0->dim[axis] == shape1->dim[axis]) {
            output_shape->dim[axis] = shape0->dim[axis];
        } else {
            LOG_error("cann't broadcast matmul inputs\n");
        }
    }
    op->output_tensors[0].shape.dim[size - 2] = m;
    op->output_tensors[0].shape.dim[size - 1] = n;
    return true;
}

static bool op_infer_shape_roipooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    uint32_t pooled_height, pooled_width;
    CHECK(op_setting_single_get(op, SETTING_ROIPOOLING_POOLED_HEIGHT, dtUINT32, &pooled_height));
    CHECK(op_setting_single_get(op, SETTING_ROIPOOLING_POOLED_WIDTH, dtUINT32, &pooled_width));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim = 0, w_dim = 0;
    if (1 == c_dim) {
        h_dim = 2;
        w_dim = 3;
    } else {
        h_dim = 1;
        w_dim = 2;
    }

    CHECK(op->output_size <= 1);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[n_dim] = n_rois;
    output_shape.dim[h_dim] = pooled_height;
    output_shape.dim[w_dim] = pooled_width;
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_psroipooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    // CHECK_EQ(2, op->input_tensors[1]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    int output_dim = 0, group_size = 0;
    CHECK(op_setting_single_get(op, SETTING_PSROIPOOLING_OUTPUT_DIM, dtUINT32, &output_dim));
    CHECK(op_setting_single_get(op, SETTING_PSROIPOOLING_GROUP_SIZE, dtUINT32, &group_size));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim], (uint32_t)(output_dim * group_size * group_size));

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape.dim_size = 4;
    op->output_tensors[0].shape.dim[0] = group_size;
    op->output_tensors[0].shape.dim[1] = group_size;
    op->output_tensors[0].shape.dim[2] = group_size;
    op->output_tensors[0].shape.dim[3] = group_size;
    op->output_tensors[0].shape.dim[n_dim] = n_rois;
    op->output_tensors[0].shape.dim[c_dim] = output_dim;
    op->output_tensors[0].shape.batch_axis = n_dim;
    op->output_tensors[0].shape.channel_axis = c_dim;
    return true;
}

static bool op_infer_shape_psroialignpooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    // CHECK_EQ(2, op->input_tensors[1]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    int output_dim = 0, group_size = 0;
    CHECK(op_setting_single_get(op, SETTING_PSROIPOOLING_OUTPUT_DIM, dtUINT32, &output_dim));
    CHECK(op_setting_single_get(op, SETTING_PSROIPOOLING_GROUP_SIZE, dtUINT32, &group_size));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim], (uint32_t)(output_dim * group_size * group_size));

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape.dim_size = 4;
    op->output_tensors[0].shape.dim[0] = group_size;
    op->output_tensors[0].shape.dim[1] = group_size;
    op->output_tensors[0].shape.dim[2] = group_size;
    op->output_tensors[0].shape.dim[3] = group_size;
    op->output_tensors[0].shape.dim[n_dim] = n_rois;
    op->output_tensors[0].shape.dim[c_dim] = output_dim;
    op->output_tensors[0].shape.batch_axis = n_dim;
    op->output_tensors[0].shape.channel_axis = c_dim;
    return true;
}

static bool op_infer_shape_psroimaskpooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    // CHECK_EQ(2, op->input_tensors[1]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    int output_dim = 0, group_size = 0;
    CHECK(op_setting_single_get(op, SETTING_PSROIMASKPOOLING_OUTPUT_DIM, dtUINT32, &output_dim));
    CHECK(op_setting_single_get(op, SETTING_PSROIMASKPOOLING_GROUP_SIZE, dtUINT32, &group_size));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    CHECK_EQ(
        op->input_tensors[0]->shape.dim[c_dim], (uint32_t)(output_dim * group_size * group_size));

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape.dim_size = 4;
    op->output_tensors[0].shape.dim[0] = group_size;
    op->output_tensors[0].shape.dim[1] = group_size;
    op->output_tensors[0].shape.dim[2] = group_size;
    op->output_tensors[0].shape.dim[3] = group_size;
    op->output_tensors[0].shape.dim[n_dim] = n_rois;
    op->output_tensors[0].shape.dim[c_dim] = output_dim;
    op->output_tensors[0].shape.batch_axis = n_dim;
    op->output_tensors[0].shape.channel_axis = c_dim;
    return true;
}

static bool op_infer_shape_roialignpooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    // CHECK_EQ(2, op->input_tensors[1]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    int sample_num = 0;
    int pooled_h = 0, pooled_w = 0;
    CHECK(op_setting_single_get(op, SETTING_ROIALIGNPOOLING_POOLED_HEIGHT, dtUINT32, &pooled_h));
    CHECK(op_setting_single_get(op, SETTING_ROIALIGNPOOLING_POOLED_WIDTH, dtUINT32, &pooled_w));
    CHECK(op_setting_single_get(op, SETTING_ROIALIGNPOOLING_SAMPLE_NUM, dtUINT32, &sample_num));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape.dim_size = 4;
    int i, j;
    for (i = 0, j = 0; i < 4; i++) {
        if (i == n_dim) {
            op->output_tensors[0].shape.dim[i] = n_rois;
        } else if (i == c_dim) {
            op->output_tensors[0].shape.dim[i] = op->input_tensors[0]->shape.dim[i];
        } else {
            op->output_tensors[0].shape.dim[i] = j++ == 0 ? pooled_h : pooled_w;
        }
    }
    op->output_tensors[0].shape.batch_axis = n_dim;
    op->output_tensors[0].shape.channel_axis = c_dim;
    return true;
}

static bool op_infer_shape_podroialignpooling(op_t *op)
{
    CHECK_EQ(op->input_size, 2);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    // CHECK_EQ(2, op->input_tensors[1]->shape.dim_size);

    int n_rois = op->input_tensors[1]->shape.dim[op->input_tensors[1]->shape.batch_axis];
    int sample_num = 0;
    int pooled_h = 0, pooled_w = 0;
    CHECK(op_setting_single_get(op, SETTING_PODROIALIGNPOOLING_POOLED_HEIGHT, dtUINT32, &pooled_h));
    CHECK(op_setting_single_get(op, SETTING_PODROIALIGNPOOLING_POOLED_WIDTH, dtUINT32, &pooled_w));
    CHECK(op_setting_single_get(op, SETTING_PODROIALIGNPOOLING_SAMPLE_NUM, dtUINT32, &sample_num));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape.dim_size = 4;
    int i, j;
    for (i = 0, j = 0; i < 4; i++) {
        if (i == n_dim) {
            op->output_tensors[0].shape.dim[i] = n_rois;
        } else if (i == c_dim) {
            op->output_tensors[0].shape.dim[i] = op->input_tensors[0]->shape.dim[i];
        } else {
            op->output_tensors[0].shape.dim[i] = j++ == 0 ? pooled_h : pooled_w;
        }
    }
    op->output_tensors[0].shape.batch_axis = n_dim;
    op->output_tensors[0].shape.channel_axis = c_dim;
    return true;
}

static bool op_infer_shape_heatmap2coord(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);

    // uint32_t coord_h, coord_w;
    // bool reposition;
    // CHECK(op_setting_single_get(op, SETTING_HEATMAP2COORD_COORD_H, dtUINT32, &coord_h));
    // CHECK(op_setting_single_get(op, SETTING_HEATMAP2COORD_COORD_W, dtUINT32, &coord_w));
    // CHECK(op_setting_single_get(op, SETTING_HEATMAP2COORD_REPOSITION, dtUINT32, &reposition));

    int n_dim = op->input_tensors[0]->shape.batch_axis;
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    // int h_dim = 0, w_dim = 0;
    // while(h_dim == op->input_tensors[0]->shape.batch_axis || h_dim ==
    // op->input_tensors[0]->shape.channel_axis) {
    //     h_dim++;
    // }
    // w_dim = h_dim + 1;

    // infer the output shape
    CHECK_GE(1, op->output_size);
    int i;
    shape_t output_shape;
    output_shape.batch_axis = op->input_tensors[0]->shape.batch_axis;
    output_shape.channel_axis = 1 - output_shape.batch_axis;
    output_shape.dim_size = 2;
    for (i = 0; i < MAX_DIM; ++i)
        output_shape.dim[i] = 1;
    output_shape.dim[output_shape.batch_axis] = op->input_tensors[0]->shape.dim[n_dim];
    output_shape.dim[output_shape.channel_axis] = op->input_tensors[0]->shape.dim[c_dim] * 3;
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_default_bilateralslice(op_t *op)
{
    CHECK_EQ(op->input_size, 3);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(op->input_tensors[1]->dtype, op->input_tensors[2]->dtype);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);

    CHECK(op->output_size <= 1);
    if (op->output_size == 0) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_exchange(op_t *op)
{
    CHECK_EQ(1, op->input_size);

    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK(
        1 == op->input_tensors[0]->shape.channel_axis
        || 3 == op->input_tensors[0]->shape.channel_axis);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);
    int c_dim = op->input_tensors[0]->shape.channel_axis;
    int h_dim;
    int w_dim;
    if (1 == c_dim) {
        /* input:  n c h w */
        h_dim = 2;
        w_dim = 3;
    } else {
        /* input:  n h w c */
        h_dim = 1;
        w_dim = 2;
    }
    int n_dim = op->input_tensors[0]->shape.batch_axis;

    CHECK_GE(1, op->output_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    op->output_tensors[0].shape.dim[n_dim] = op->input_tensors[0]->shape.dim[w_dim];
    op->output_tensors[0].shape.dim[c_dim] = op->input_tensors[0]->shape.dim[n_dim];
    op->output_tensors[0].shape.dim[h_dim] = op->input_tensors[0]->shape.dim[c_dim];
    op->output_tensors[0].shape.dim[w_dim] = op->input_tensors[0]->shape.dim[h_dim];
    return true;
}

static bool op_infer_shape_shufflechannel_operator(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);

    int group;
    CHECK(op_setting_single_get(op, SETTING_SHUFFLECHANNEL_GROUP, dtUINT32, &group));
    int group_row = group;
    int group_column = op->input_tensors[0]->shape.dim[1] / group_row;
    int channel = op->input_tensors[0]->shape.dim[1];
    CHECK_EQ(channel, group_column * group_row);

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_pad(op_t *op)
{
    int32_t *var;
    size_t len, i;
    CHECK(op_setting_array_get(op, SETTING_PAD_PADS, dtINT32, &len, &var));
    shape_t output_shape = op->input_tensors[0]->shape;
    CHECK(output_shape.dim_size * 2 == len);
    size_t dim_size = output_shape.dim_size;
    for (i = 0; i < len; ++i) {
        output_shape.dim[i % dim_size] += var[i];
    }
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_reduce_operator(op_t *op)
{
    int32_t *var, keepdims;
    int32_t flag[MAX_DIM] = { 0 };
    size_t len, i, d;
    CHECK(op_setting_array_get(op, SETTING_REDUCE_AXES, dtINT32, &len, &var));
    CHECK(op_setting_single_get(op, SETTING_REDUCE_KEEPDIMS, dtINT32, &keepdims));
    shape_t out_shape = op->input_tensors[0]->shape;
    size_t dim_size = op->input_tensors[0]->shape.dim_size;
    for (i = 0; i < len; ++i) {
        if (var[i] < 0)
            var[i] = dim_size + var[i];
        flag[var[i]] = 1;
    }

    if (1 == keepdims) {
        if (0 != len) {
            for (i = 0; i < dim_size; ++i) {
                if (0 != flag[i])
                    out_shape.dim[i] = 1;
                else
                    out_shape.dim[i] = op->input_tensors[0]->shape.dim[i];
            }
        } else {
            for (i = 0; i < dim_size; ++i) {
                out_shape.dim[i] = 1;
            }
        }
    } else {
        if (0 != len) {
            for (i = 0, d = 0; i < dim_size; ++i) {
                if (0 != flag[i])
                    continue;
                out_shape.dim[d++] = op->input_tensors[0]->shape.dim[i];
            }
            out_shape.dim_size = d;
        } else {
            out_shape.dim_size = 1;
            out_shape.dim[0] = 1;
        }
    }
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
        op->output_tensors[0].shape.channel_axis = op->input_tensors[0]->shape.channel_axis;
    }
    op->output_tensors[0].shape = out_shape;
    return true;
}

static bool op_infer_shape_instancenorm(op_t *op)
{
    CHECK_EQ(3, op->input_size);
    CHECK(1 >= op->output_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_correlation1d(op_t *op)
{
    CHECK_EQ(2, op->input_size);
    CHECK(1 >= op->output_size);

    int max_disp;
    CHECK(op_setting_single_get(op, SETTING_CORRELATION1D_MAX_DISPLACEMENT, dtINT32, &max_disp));

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    op->output_tensors[0].shape.dim[1] = max_disp + 1;
    return true;
}

static bool op_infer_shape_gather(op_t *op)
{
    CHECK_EQ(2, op->input_size);
    CHECK(1 >= op->output_size);

    int axis;
    CHECK(op_setting_single_get(op, SETTING_GATHER_AXIS, dtINT32, &axis));

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    int r = op->input_tensors[0]->shape.dim_size;
    int q = op->input_tensors[1]->shape.dim_size;
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    op->output_tensors[0].shape.dim_size += q - 1;
    int i = 0;
    axis = axis >= 0 ? axis : axis + r;

    for (i = axis; i < axis + q; i++) {
        op->output_tensors[0].shape.dim[i] = op->input_tensors[1]->shape.dim[i - axis];
    }
    for (; i < r + q - 1; i++) {
        op->output_tensors[0].shape.dim[i] = op->input_tensors[0]->shape.dim[i - q + 1];
    }

    shape_t shape = op->output_tensors[0].shape;
    return true;
}

static bool op_infer_shape_argmax(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);

    int axis, keepdims;
    CHECK(op_setting_single_get(op, SETTING_ARGMAX_AXIS, dtINT32, &axis));
    CHECK(op_setting_single_get(op, SETTING_ARGMAX_KEEPDIMS, dtINT32, &keepdims));

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    axis = axis >= 0 ? axis : axis + op->output_tensors[0].shape.dim_size;

    if (keepdims != 0) {
        op->output_tensors[0].shape.dim[axis] = 1;
    } else {
        int i;
        for (i = axis + 1; i < op->output_tensors[0].shape.dim_size; i++) {
            op->output_tensors[0].shape.dim[i - 1] = op->output_tensors[0].shape.dim[i];
        }

        op->output_tensors[0].shape.dim_size -= 1;
        if (op->output_tensors[0].shape.dim_size == 0) {
            op->output_tensors[0].shape.dim_size = 1;
            op->output_tensors[0].shape.dim[0] = 1;
        }
    }
    return true;
}

static bool op_infer_shape_gridsample(op_t *op)
{
    // (N, C, Hin, Win) x (N, Hout, Wout, 2) -> (N, C, Hout, Wout)
    // (N, C, Din, Hin, Win) x (N, Dout, Hout, Wout, 3) -> (N, C, Dout, Hout, Wout)
    CHECK_EQ(2, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK_EQ(op->input_tensors[0]->shape.dim_size, op->input_tensors[1]->shape.dim_size);

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    int i;
    for (i = 2; i < op->output_tensors[0].shape.dim_size; i++) {
        op->output_tensors[0].shape.dim[i] = op->input_tensors[1]->shape.dim[i - 1];
    }
    return true;
}

static bool op_infer_shape_unfold(op_t *op)
{
    // (N, C, H, W) -> (N, C*Kh*Kw, L)
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);

    uint32_t kernel_h = 0, kernel_w = 0;
    uint32_t pad_h = 0, pad_w = 0;
    uint32_t stride_h = 0, stride_w = 0;
    uint32_t hole_h = 0, hole_w = 0;

    CHECK(op_setting_single_get(op, SETTING_UNFOLD_PAD_H, dtUINT32, &pad_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_PAD_W, dtUINT32, &pad_w));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_KERNEL_H, dtUINT32, &kernel_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_KERNEL_W, dtUINT32, &kernel_w));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_STRIDE_H, dtUINT32, &stride_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_STRIDE_W, dtUINT32, &stride_w));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_HOLE_H, dtUINT32, &hole_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_HOLE_W, dtUINT32, &hole_w));

    CHECK_EQ(0, op->input_tensors[0]->shape.batch_axis);
    CHECK_EQ(4, op->input_tensors[0]->shape.dim_size);

    uint32_t kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
    uint32_t kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);

    uint32_t oh = (op->input_tensors[0]->shape.dim[2] + pad_h * 2 - kernel_h_eff) / stride_h + 1;
    uint32_t ow = (op->input_tensors[0]->shape.dim[3] + pad_w * 2 - kernel_w_eff) / stride_w + 1;

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim_size = 3;
    output_shape.dim[1] = op->input_tensors[0]->shape.dim[1] * kernel_h * kernel_w;
    output_shape.dim[2] = oh * ow;

    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    return true;
}

static bool op_infer_shape_topk(op_t *op)
{
    int axis, k;
    CHECK(op_setting_single_get(op, SETTING_TOPK_AXIS, dtINT32, &axis));
    CHECK(op_setting_single_get(op, SETTING_TOPK_K, dtINT32, &k));

    CHECK_GE(k, 1);

    if (axis < 0) {
        axis += op->input_tensors[0]->shape.dim_size;
    }

    shape_t output_shape = op->input_tensors[0]->shape;
    output_shape.dim[axis] = k;

    if (0 == op->output_size) {
        op->output_size = 2;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
        op->output_tensors[1].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[1].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = output_shape;
    op->output_tensors[1].shape = output_shape;
    return true;
}

static bool op_infer_shape_lstm(op_t *op)
{
    /*
     */
    CHECK_LE(3, op->input_size);
    CHECK_NE(4, op->input_size);
    CHECK_GE(5, op->input_size);

    uint32_t hidden_size;
    uint32_t direction;
    uint32_t input_forget;
    float clip;
    uint32_t clip_exist;
    uint32_t activation_f, activation_g, activation_h;
    uint32_t activation_alpha_f, activation_alpha_g, activation_alpha_h;
    uint32_t activation_beta_f, activation_beta_g, activation_beta_h;
    uint32_t output_size;
    CHECK(op_setting_single_get(op, SETTING_LSTM_HIDDEN_SIZE, dtINT32, &hidden_size));
    CHECK(op_setting_single_get(op, SETTING_LSTM_DIRECTION, dtINT32, &direction));
    CHECK(op_setting_single_get(op, SETTING_LSTM_INPUT_FORGET, dtINT32, &input_forget));
    CHECK(op_setting_single_get(op, SETTING_LSTM_CLIP, dtFLOAT32, &clip));
    CHECK(op_setting_single_get(op, SETTING_LSTM_CLIP_EXIST, dtINT32, &clip_exist));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_F, dtINT32, &activation_f));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_G, dtINT32, &activation_g));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_H, dtINT32, &activation_h));
    CHECK(
        op_setting_single_get(op, SETTING_LSTM_ACTIVATION_ALPHA_F, dtFLOAT32, &activation_alpha_f));
    CHECK(
        op_setting_single_get(op, SETTING_LSTM_ACTIVATION_ALPHA_G, dtFLOAT32, &activation_alpha_g));
    CHECK(
        op_setting_single_get(op, SETTING_LSTM_ACTIVATION_ALPHA_G, dtFLOAT32, &activation_alpha_h));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_BETA_F, dtFLOAT32, &activation_beta_f));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_BETA_G, dtFLOAT32, &activation_beta_g));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_BETA_H, dtFLOAT32, &activation_beta_h));
    CHECK(op_setting_single_get(op, SETTING_LSTM_OUTPUT_SIZE, dtINT32, &output_size));

    if (clip > 1e-8 || clip < -1e-8) {
        CHECK_EQ(0, clip_exist);
    }

    uint32_t num_directions;
    if (direction == LSTM_DIRECTION_BIDIRECTIONAL)
        num_directions = 2;
    else
        num_directions = 1;

    shape_t input_shape_0 = op->input_tensors[0]->shape;
    shape_t input_shape_1 = op->input_tensors[1]->shape;
    shape_t input_shape_2 = op->input_tensors[2]->shape;

    CHECK_EQ(3, input_shape_0.dim_size);
    uint32_t seq_length = input_shape_0.dim[0];
    uint32_t batch_size = input_shape_0.dim[1];
    uint32_t input_size = input_shape_0.dim[2];

    CHECK_EQ(3, input_shape_1.dim_size);
    CHECK_EQ(num_directions, input_shape_1.dim[0]);
    CHECK_EQ(hidden_size * 4, input_shape_1.dim[1]);
    CHECK_EQ(input_size, input_shape_1.dim[2]);

    CHECK_EQ(3, input_shape_2.dim_size);
    CHECK_EQ(num_directions, input_shape_2.dim[0]);
    CHECK_EQ(hidden_size * 4, input_shape_2.dim[1]);
    CHECK_EQ(hidden_size, input_shape_2.dim[2]);

    if (op->input_size >= 4) {
        shape_t input_shape_3 = op->input_tensors[3]->shape;
        CHECK_EQ(2, input_shape_3.dim_size);
        CHECK_EQ(num_directions, input_shape_3.dim[0]);
        CHECK_EQ(hidden_size * 8, input_shape_3.dim[1]);
    }

    if (op->input_size >= 5) {
        shape_t input_shape_4 = op->input_tensors[4]->shape;
        CHECK_EQ(1, input_shape_4.dim_size);
        CHECK_EQ(batch_size, input_shape_4.dim[0]);
    }

    if (0 == op->output_size) {
        op->output_size = output_size;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
    }

    if (op->output_size >= 1) {
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;

        shape_t output_shape_0;
        output_shape_0.dim_size = 4;
        output_shape_0.batch_axis = 2;
        output_shape_0.channel_axis = 3;
        output_shape_0.dim[0] = seq_length;
        output_shape_0.dim[1] = num_directions;
        output_shape_0.dim[2] = batch_size;
        output_shape_0.dim[3] = hidden_size;

        op->output_tensors[0].shape = output_shape_0;
    }

    if (op->output_size >= 2) {
        op->output_tensors[1].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[1].dtype = op->input_tensors[0]->dtype;

        shape_t output_shape_1;
        output_shape_1.dim_size = 3;
        output_shape_1.batch_axis = 1;
        output_shape_1.channel_axis = 2;
        output_shape_1.dim[0] = num_directions;
        output_shape_1.dim[1] = batch_size;
        output_shape_1.dim[2] = hidden_size;

        op->output_tensors[1].shape = output_shape_1;
    }

    if (op->output_size >= 3) {
        op->output_tensors[2].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[2].dtype = op->input_tensors[0]->dtype;

        shape_t output_shape_2;
        output_shape_2.dim_size = 3;
        output_shape_2.batch_axis = 1;
        output_shape_2.channel_axis = 2;
        output_shape_2.dim[0] = num_directions;
        output_shape_2.dim[1] = batch_size;
        output_shape_2.dim[2] = hidden_size;

        op->output_tensors[2].shape = output_shape_2;
    }

    return true;
}

static bool op_infer_shape_clip(op_t *op)
{
    CHECK((op->input_size == 1) || (op->input_size == 3));
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_cast(op_t *op)
{
    CHECK_EQ(1, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
    }
    CHECK(op_setting_single_get(op, SETTING_CAST_DTYPE, dtUINT32, &op->output_tensors[0].dtype));
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_scatternd(op_t *op)
{
    CHECK_EQ(3, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
        op->output_tensors[0].dtype = op->input_tensors[0]->dtype;
    }
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_clip_cast(op_t *op)
{
    CHECK_EQ(3, op->input_size);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
    }
    CHECK(op_setting_single_get(op, SETTING_CAST_DTYPE, dtUINT32, &op->output_tensors[0].dtype));
    op->output_tensors[0].shape = op->input_tensors[0]->shape;
    return true;
}

static bool op_infer_shape_add_div_clip_cast(op_t *op)
{
    CHECK_EQ(5, op->input_size);
    CHECK_EQ(op->input_tensors[0]->dtype, op->input_tensors[1]->dtype);
    CHECK_EQ(op->input_tensors[1]->dtype, op->input_tensors[2]->dtype);
    CHECK(1 >= op->output_size);
    CHECK(1 <= op->input_tensors[0]->shape.dim_size);
    if (0 == op->output_size) {
        op->output_size = 1;
        op->output_tensors = (tensor_t *)malloc(sizeof(tensor_t) * op->output_size);
        memset(op->output_tensors, 0, sizeof(tensor_t) * op->output_size);
        op->output_tensors[0].mem = mem_new(op->input_tensors[0]->mem->tp);
    }
    CHECK(op_setting_single_get(op, SETTING_CAST_DTYPE, dtUINT32, &op->output_tensors[0].dtype));
    size_t dim1 = op->input_tensors[0]->shape.dim_size;
    size_t dim2 = op->input_tensors[1]->shape.dim_size;
    size_t dim = dim1 > dim2 ? dim1 : dim2;
    shape_t shape_0 = op->input_tensors[0]->shape;
    shape_t shape_1 = op->input_tensors[1]->shape;

    shape_0.dim_size = dim;
    shape_1.dim_size = dim;

    int i;
    for (i = 0; i < dim1; ++i) {
        shape_0.dim[dim - 1 - i] = shape_0.dim[dim1 - 1 - i];
    }
    for (; i < dim; ++i) {
        shape_0.dim[dim - 1 - i] = 1;
    }
    for (i = 0; i < dim2; ++i) {
        shape_1.dim[dim - 1 - i] = shape_1.dim[dim2 - 1 - i];
    }
    for (; i < dim; ++i) {
        shape_1.dim[dim - 1 - i] = 1;
    }

    op->output_tensors[0].shape = shape_0;
    for (i = 0; i < dim; ++i) {
        if (shape_0.dim[i] == shape_1.dim[i])
            op->output_tensors[0].shape.dim[i] = shape_0.dim[i];
        else if (shape_0.dim[i] == 1)
            op->output_tensors[0].shape.dim[i] = shape_1.dim[i];
        else if (shape_1.dim[i] == 1)
            op->output_tensors[0].shape.dim[i] = shape_0.dim[i];
        else
            LOG(error, "cannot broadcast %d dim: %d vs %d\n", i, shape_0.dim[i], shape_1.dim[i]);
    }
    return true;
}

const op_tp_t op_pad_tp = {
    .op_tp_code = OP_PAD,
    .name = "pad",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_pad,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PAD_MODE, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PAD_VALUE, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_REPEATED(SETTING_PAD_PADS, dtINT32), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_sqrt_tp = { .op_tp_code = OP_SQRT,
                             .name = "sqrt",
                             .min_input_size = 1,
                             .max_input_size = 1,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_unary_operator,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_abs_tp = { .op_tp_code = OP_ABS,
                            .name = "abs",
                            .min_input_size = 1,
                            .max_input_size = 1,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_unary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_floor_tp = { .op_tp_code = OP_FLOOR,
                              .name = "floor",
                              .min_input_size = 1,
                              .max_input_size = 1,
                              .min_output_size = 1,
                              .max_output_size = 1,
                              .infer_output_func = op_infer_shape_unary_operator,
                              .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_add_tp = { .op_tp_code = OP_ADD,
                            .name = "add",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_sub_tp = { .op_tp_code = OP_SUB,
                            .name = "sub",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_mul_tp = { .op_tp_code = OP_MUL,
                            .name = "mul",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_div_tp = { .op_tp_code = OP_DIV,
                            .name = "div",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_pow_tp = { .op_tp_code = OP_POW,
                            .name = "pow",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_log_tp = { .op_tp_code = OP_LOG,
                            .name = "log",
                            .min_input_size = 1,
                            .max_input_size = 1,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_unary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_lpnormalization_tp = {
    .op_tp_code = OP_LPNORMALIZATION,
    .name = "lpnormalization",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_unary_operator,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LPNORMALIZATION_P, dtINT32, 2),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LPNORMALIZATION_AXIS, dtINT32, -1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_exp_tp = { .op_tp_code = OP_EXP,
                            .name = "exp",
                            .min_input_size = 1,
                            .max_input_size = 1,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_unary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_conv_2d_tp = {
.op_tp_code = OP_CONV_2D,
.name = "conv_2d",
.min_input_size = 2,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_conv_2d,
.constraints = {
/* num_output */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_NUM_OUTPUT, dtUINT32),
/* kernel size */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_H, dtUINT32),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_W, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_W, dtUINT32, 0),
/* stride */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_W, dtUINT32, 1),
/* dilation */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_HOLE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_HOLE_W, dtUINT32, 1),
/* group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_GROUP, dtUINT32, 1),
/* relu */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_RELU_FLAG, dtBOOL, false),

OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_conv_nd_tp = {
.op_tp_code = OP_CONV_ND,
.name = "conv_nd",
.min_input_size = 2,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_conv_nd,
.constraints = {
/* num_output */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_ND_NUM_OUTPUT, dtUINT32),
/* kernel size */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_CONV_ND_KERNEL, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_CONV_ND_PAD, dtUINT32),
/* stride */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_CONV_ND_STRIDE, dtUINT32),
/* dilation */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_CONV_ND_HOLE, dtUINT32),
/* group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_ND_GROUP, dtUINT32, 1),
/* relu */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_ND_RELU_FLAG, dtBOOL, false),

OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_quant_dequant_tp = {
.op_tp_code = OP_QUANT_DEQUANT,
.name = "quant_dequant",
.min_input_size = 2,
.max_input_size = 2,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_quant_dequant,
.constraints = {
/* q_min */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_DEQUANT_QMIN, dtINT32),
/* q_max */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_DEQUANT_QMAX, dtINT32),
/* scale */
OP_SETTING_CONSTRAINT_REPEATED(SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32),

OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_deform_conv_2d_tp = {
.op_tp_code = OP_DEFORM_CONV_2D,
.name = "deform_conv_2d",
.min_input_size = 3,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_deform_conv_2d,
.constraints = {
/* num_output */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_NUM_OUTPUT, dtUINT32),
/* kernel size */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_H, dtUINT32),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_W, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_W, dtUINT32, 0),
/* stride */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_W, dtUINT32, 1),
/* group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_GROUP, dtUINT32, 1),
/* deform_group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_DEFORM_GROUP, dtUINT32, 1),
/* hole */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_HOLE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_HOLE_W, dtUINT32, 1),

OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_deconv_2d_tp = {
.op_tp_code = OP_DECONV_2D,
.name = "deconv_2d",
.min_input_size = 2,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_deconv_2d,
.constraints = {
/* num_output */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_NUM_OUTPUT, dtUINT32),
/* kernel size */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_H, dtUINT32),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_W, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_W, dtUINT32, 0),
/* stride */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_W, dtUINT32, 1),
/* group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_GROUP, dtUINT32, 1),

OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_correlation_tp = {
    .op_tp_code = OP_CORRELATION,
    .name = "correlation",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_correlation,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CORRELATION_GROUPS, dtUINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_relu_tp = { .op_tp_code = OP_RELU,
                             .name = "relu",
                             .min_input_size = 1,
                             .max_input_size = 1,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_unary_operator,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_relu6_tp = { .op_tp_code = OP_RELU6,
                              .name = "relu6",
                              .min_input_size = 1,
                              .max_input_size = 1,
                              .min_output_size = 1,
                              .max_output_size = 1,
                              .infer_output_func = op_infer_shape_unary_operator,
                              .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_tanh_tp = { .op_tp_code = OP_TANH,
                             .name = "tanh",
                             .min_input_size = 1,
                             .max_input_size = 1,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_unary_operator,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_prelu_tp = {
    .op_tp_code = OP_PRELU,
    .name = "prelu",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_prelu,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PRELU_SHARE, dtBOOL), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_lrn_tp = {
    .op_tp_code = OP_LRN,
    .name = "lrn",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_unary_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LRN_LOCAL_SIZE, dtUINT32),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LRN_ALPHA, dtFLOAT32),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LRN_BETA, dtFLOAT32),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LRN_K, dtFLOAT32),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LRN_NORM_REGION, dtUINT32),

                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_sigmoid_tp = { .op_tp_code = OP_SIGMOID,
                                .name = "sigmoid",
                                .min_input_size = 1,
                                .max_input_size = 1,
                                .min_output_size = 1,
                                .max_output_size = 1,
                                .infer_output_func = op_infer_shape_unary_operator,
                                .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_softmax_tp = {
.op_tp_code = OP_SOFTMAX,
.name = "softmax",
.min_input_size = 1,
.max_input_size = 1,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_unary_operator,
.constraints = {
#ifdef __leon__
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_SOFTMAX_AXIS, dtUINT32, 3),
#else
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_SOFTMAX_AXIS, dtUINT32, 1),
#endif
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_eltwise_tp = {
    .op_tp_code = OP_ELTWISE,
    .name = "eltwise",
    .min_input_size = 2,
    .max_input_size = 0xffff,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_eltwise,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ELTWISE_OPERATION, dtUINT32, SETTING_ELTWISE_OP_SUM),
       OP_SETTING_CONSTRAINT_REPEATED(SETTING_ELTWISE_COEFF, dtFLOAT32),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_bn_tp = {
    .op_tp_code = OP_BN,
    .name = "bn",
    .min_input_size = 3,
    .max_input_size = 5,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_bn,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_BN_EPS, dtFLOAT32), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_batchnorm_tp = {
    .op_tp_code = OP_BATCHNORM,
    .name = "batchnorm",
    .min_input_size = 1,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_batchnorm,
    .constraints = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_BATCHNORM_EPS, dtFLOAT32),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_ip_tp = {
    .op_tp_code = OP_IP,
    .name = "ip",
    .min_input_size = 2,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_default_ip,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_IP_NUM_OUTPUT, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_IP_RELU_FLAG, dtBOOL, false),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_IP_AXIS, dtINT32, 1), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_concat_tp = {
    .op_tp_code = OP_CONCAT,
    .name = "concat",
    .min_input_size = 2,
    .max_input_size = 0xffff,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_concat,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONCAT_AXIS, dtUINT32), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_pool_tp = {
    .op_tp_code = OP_POOL,
    .name = "pool",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_pool,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_POOL_KERNEL_H, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_POOL_KERNEL_W, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_METHOD, dtUINT32, SETTING_POOL_MAX),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_PAD_H, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_PAD_W, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_STRIDE_H, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_STRIDE_W, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_POOL_CEIL_MODE, dtBOOL, true),

       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_interp_tp = {
    .op_tp_code = OP_INTERP,
    .name = "interp",
    .min_input_size = 1,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_default_interp,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_HEIGHT, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_WIDTH, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_ZOOM_FACTOR, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_SHRINK_FACTOR, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_PAD_BEG, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_PAD_END, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INTERP_TYPE, dtUINT32, SETTING_INTERP_BILINEAR),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_slice_tp = {
    .op_tp_code = OP_SLICE,
    .name = "slice",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 0xffff,
    .infer_output_func = op_infer_shape_slice,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_SLICE_AXIS, dtUINT32),
       OP_SETTING_CONSTRAINT_REPEATED(SETTING_SLICE_POINT, dtUINT32), OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_scale_tp = {
    .op_tp_code = OP_SCALE,
    .name = "scale",
    .min_input_size = 2,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_scale,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_SCALE_BIAS_TERM, dtBOOL, false),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reshape_tp = {
    .op_tp_code = OP_RESHAPE,
    .name = "reshape",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reshape,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_RESHAPE_DIMS, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_RESHAPE_AXIS, dtINT32, 0),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_RESHAPE_NUM_AXES, dtINT32, -1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_subpixel_tp = {
    .op_tp_code = OP_SUBPIXEL,
    .name = "subpixel",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_subpixel,
    .constraints = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_SUBPIXEL_METHOD, dtUINT8),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_SUBPIXEL_SAMPLE, dtUINT32),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_transpose_tp = {
    .op_tp_code = OP_TRANSPOSE,
    .name = "transpose",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_transpose,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_TRANSPOSE_DIMS, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_TRANSPOSE_EXGAXIS, dtBOOL, false),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_matmul_tp = {
    .op_tp_code = OP_MATMUL,
    .name = "matmul",
    .min_input_size = 2,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_matmul,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_MATMUL_RELU_FLAG, dtBOOL, false),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_conv_2d_wino_tp = {
.op_tp_code = OP_CONV_2D_WINO,
.name = "conv_2d_wino",
.min_input_size = 2,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_conv_2d,
.constraints = {
/* num_output */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_NUM_OUTPUT, dtUINT32),
/* kernel size */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_H, dtUINT32),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CONV_2D_KERNEL_W, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_PAD_W, dtUINT32, 0),
/* stride */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_STRIDE_W, dtUINT32, 0),
/* group */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CONV_2D_GROUP, dtUINT32, 1),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_roipooling_tp = {
    .op_tp_code = OP_ROIPOOLING,
    .name = "roipooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_roipooling,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ROIPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_ROIPOOLING_POOLED_HEIGHT, dtUINT32),
                    OP_SETTING_CONSTRAINT_REQUIRED(SETTING_ROIPOOLING_POOLED_WIDTH, dtUINT32),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_psroipooling_tp = {
    .op_tp_code = OP_PSROIPOOLING,
    .name = "psroipooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_psroipooling,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIPOOLING_OUTPUT_DIM, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIPOOLING_GROUP_SIZE, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIPOOLING_SAMPLE_NUM, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_psroialignpooling_tp = {
    .op_tp_code = OP_PSROIALIGNPOOLING,
    .name = "psroialignpooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_psroialignpooling,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIALIGNPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIALIGNPOOLING_OUTPUT_DIM, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIALIGNPOOLING_GROUP_SIZE, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIALIGNPOOLING_SAMPLE_NUM, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_roialignpooling_tp = {
    .op_tp_code = OP_ROIALIGNPOOLING,
    .name = "roialignpooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_roialignpooling,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ROIALIGNPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_ROIALIGNPOOLING_POOLED_HEIGHT, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_ROIALIGNPOOLING_POOLED_WIDTH, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ROIALIGNPOOLING_SAMPLE_NUM, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_podroialignpooling_tp = {
    .op_tp_code = OP_PODROIALIGNPOOLING,
    .name = "podroialignpooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_podroialignpooling,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PODROIALIGNPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PODROIALIGNPOOLING_POOLED_HEIGHT, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PODROIALIGNPOOLING_POOLED_WIDTH, dtUINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PODROIALIGNPOOLING_SAMPLE_NUM, dtUINT32, 1),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_psroimaskpooling_tp = {
    .op_tp_code = OP_PSROIMASKPOOLING,
    .name = "psroimaskpooling",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_psroimaskpooling,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIMASKPOOLING_SPATIAL_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIMASKPOOLING_ROI_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_PSROIMASKPOOLING_BIN_SCALE, dtFLOAT32, 1),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIMASKPOOLING_GROUP_SIZE, dtUINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_PSROIMASKPOOLING_OUTPUT_DIM, dtUINT32),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_heatmap2coord_tp = {
    .op_tp_code = OP_HEATMAP2COORD,
    .name = "heatmap2coord",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_heatmap2coord,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_HEATMAP2COORD_COORD_H, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_HEATMAP2COORD_COORD_W, dtUINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_HEATMAP2COORD_REPOSITION, dtBOOL, false),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_exchange_tp = { .op_tp_code = OP_EXCHANGE,
                                 .name = "exchange",
                                 .min_input_size = 1,
                                 .max_input_size = 1,
                                 .min_output_size = 1,
                                 .max_output_size = 1,
                                 .infer_output_func = op_infer_shape_exchange,
                                 .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_bilateralslice_tp = {
    .op_tp_code = OP_BILATERALSLICE,
    .name = "bilateralslice",
    .min_input_size = 3,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_default_bilateralslice,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_BILATERALSLICE_COE, dtUINT32, 12),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_BILATERALSLICE_OFFSET, dtBOOL, true),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_shufflechannel_tp = {
    .op_tp_code = OP_SHUFFLECHANNEL,
    .name = "shufflechannel",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_shufflechannel_operator,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_SHUFFLECHANNEL_GROUP, dtUINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_instancenorm_tp = {
    .op_tp_code = OP_INSTANCENORM,
    .name = "instancenorm",
    .min_input_size = 3,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_instancenorm,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_INSTANCENORM_EPS, dtFLOAT32, 1e-5),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reducemin_tp = {
    .op_tp_code = OP_REDUCEMIN,
    .name = "reducemin",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reducemax_tp = {
    .op_tp_code = OP_REDUCEMAX,
    .name = "reducemax",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reducemean_tp = {
    .op_tp_code = OP_REDUCEMEAN,
    .name = "reducemean",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reduceprod_tp = {
    .op_tp_code = OP_REDUCEPROD,
    .name = "reduceprod",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reducesum_tp = {
    .op_tp_code = OP_REDUCESUM,
    .name = "reducesum",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_reducel2_tp = {
    .op_tp_code = OP_REDUCEL2,
    .name = "reducel2",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_reduce_operator,
    .constraints = {OP_SETTING_CONSTRAINT_REPEATED(SETTING_REDUCE_AXES, dtINT32),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_REDUCE_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_correlation1d_tp = {
    .op_tp_code = OP_CORRELATION1D,
    .name = "correlation1d",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_correlation1d,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CORRELATION1D_MAX_DISPLACEMENT, dtINT32),
       OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CORRELATION1D_KERNEL_SIZE, dtINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CORRELATION1D_SINGLE_DIRECTION, dtINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_CORRELATION1D_PAD, dtINT32, 0),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_gather_tp = {
    .op_tp_code = OP_GATHER,
    .name = "gather",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_gather,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_GATHER_AXIS, dtINT32, 0),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_argmax_tp = {
    .op_tp_code = OP_ARGMAX,
    .name = "argmax",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_argmax,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ARGMAX_AXIS, dtINT32, 0),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ARGMAX_KEEPDIMS, dtINT32, 1),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ARGMAX_SELECT_LAST_INDEX, dtINT32, 0),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_gridsample_tp = {
    .op_tp_code = OP_GRIDSAMPLE,
    .name = "gridsample",
    .min_input_size = 2,
    .max_input_size = 2,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_gridsample,
    .constraints
    = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_GRIDSAMPLE_MODE, dtINT32, GRIDSAMPLE_MODE_BILINEAR),
       OP_SETTING_CONSTRAINT_OPTIONAL(
            SETTING_GRIDSAMPLE_PADDING_MODE, dtINT32, GRIDSAMPLE_PADDING_ZEROS),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_GRIDSAMPLE_ALIGN_CORNERS, dtBOOL, 0),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_unfold_tp = {
.op_tp_code = OP_UNFOLD,
.name = "unfold",
.min_input_size = 1,
.max_input_size = 1,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_unfold,
.constraints = {
/* kernel size */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_UNFOLD_KERNEL_H, dtUINT32),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_UNFOLD_KERNEL_W, dtUINT32),
/* pad */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_PAD_H, dtUINT32, 0),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_PAD_W, dtUINT32, 0),
/* stride */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_STRIDE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_STRIDE_W, dtUINT32, 1),
/* dilation */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_HOLE_H, dtUINT32, 1),
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_UNFOLD_HOLE_W, dtUINT32, 1),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_topk_tp = {
.op_tp_code = OP_TOPK,
.name = "topk",
.min_input_size = 1,
.max_input_size = 2,
.min_output_size = 2,
.max_output_size = 2,
.infer_output_func = op_infer_shape_topk,
.constraints = {
/* dilation */
OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_TOPK_AXIS, dtINT32, -1),
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_TOPK_K, dtINT32),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_lstm_tp = {
    .op_tp_code = OP_LSTM,
    .name = "lstm",
    .min_input_size = 3,
    .max_input_size = 5,
    .min_output_size = 0,
    .max_output_size = 3,
    .infer_output_func = op_infer_shape_lstm,
    .constraints
    = {OP_SETTING_CONSTRAINT_REQUIRED(SETTING_LSTM_HIDDEN_SIZE, dtINT32),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_DIRECTION, dtINT32, LSTM_DIRECTION_FORWARD),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_INPUT_FORGET, dtINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_CLIP, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_CLIP_EXIST, dtINT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_F, dtINT32, LSTM_ACTIVATION_SIGMOID),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_ALPHA_F, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_BETA_F, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_G, dtINT32, LSTM_ACTIVATION_TANH),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_ALPHA_G, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_BETA_G, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_H, dtINT32, LSTM_ACTIVATION_TANH),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_ALPHA_H, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_ACTIVATION_BETA_H, dtFLOAT32, 0),
       OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_LSTM_OUTPUT_SIZE, dtINT32, 3),
       OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_hardsigmoid_tp = {
    .op_tp_code = OP_HARDSIGMOID,
    .name = "hardsigmoid",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_unary_operator,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_HARDSIGMOID_ALPHA, dtFLOAT32, 0.2),
                    OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_HARDSIGMOID_BETA, dtFLOAT32, 0.5),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_erf_tp = { .op_tp_code = OP_ERF,
                            .name = "erf",
                            .min_input_size = 1,
                            .max_input_size = 1,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_unary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_clip_tp = { .op_tp_code = OP_CLIP,
                             .name = "clip",
                             .min_input_size = 1,
                             .max_input_size = 3,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_clip,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_cast_tp = {
.op_tp_code = OP_CAST,
.name = "cast",
.min_input_size = 1,
.max_input_size = 1,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_cast,
.constraints = {
/* Min & Max */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CAST_DTYPE, dtUINT32),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_hswish_tp = { .op_tp_code = OP_HSWISH,
                               .name = "hswish",
                               .min_input_size = 1,
                               .max_input_size = 1,
                               .min_output_size = 1,
                               .max_output_size = 1,
                               .infer_output_func = op_infer_shape_unary_operator,
                               .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_clip_cast_tp = {
.op_tp_code = OP_CLIP_CAST,
.name = "clip_cast",
.min_input_size = 3,
.max_input_size = 3,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_clip_cast,
.constraints = {
/* Min & Max */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CAST_DTYPE, dtUINT32),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_add_div_clip_cast_tp = {
.op_tp_code = OP_ADD_DIV_CLIP_CAST,
.name = "add_div_clip_cast",
.min_input_size = 5,
.max_input_size = 5,
.min_output_size = 1,
.max_output_size = 1,
.infer_output_func = op_infer_shape_add_div_clip_cast,
.constraints = {
/* Min & Max */
OP_SETTING_CONSTRAINT_REQUIRED(SETTING_CAST_DTYPE, dtUINT32),
OP_SETTING_CONSTRAINT_END()
}
};

const op_tp_t op_scatternd_tp = {
    .op_tp_code = OP_SCATTERND,
    .name = "scatternd",
    .min_input_size = 3,
    .max_input_size = 3,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_scatternd,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_SCATTERND_REDUCTION, dtSTR, ""),
                    OP_SETTING_CONSTRAINT_END()}
};

const op_tp_t op_min_tp = { .op_tp_code = OP_MIN,
                            .name = "min",
                            .min_input_size = 2,
                            .max_input_size = 2,
                            .min_output_size = 1,
                            .max_output_size = 1,
                            .infer_output_func = op_infer_shape_binary_operator,
                            .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_less_tp = { .op_tp_code = OP_LESS,
                             .name = "less",
                             .min_input_size = 1,
                             .max_input_size = 1,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_cast,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_where_tp = { .op_tp_code = OP_WHERE,
                              .name = "where",
                              .min_input_size = 1,
                              .max_input_size = 1,
                              .min_output_size = 1,
                              .max_output_size = 1,
                              .infer_output_func = op_infer_shape_cast,
                              .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_sign_tp = { .op_tp_code = OP_SIGN,
                             .name = "sign",
                             .min_input_size = 1,
                             .max_input_size = 1,
                             .min_output_size = 1,
                             .max_output_size = 1,
                             .infer_output_func = op_infer_shape_unary_operator,
                             .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_roundto0_tp = { .op_tp_code = OP_ROUNDTO0,
                                 .name = "roundto0",
                                 .min_input_size = 1,
                                 .max_input_size = 1,
                                 .min_output_size = 1,
                                 .max_output_size = 1,
                                 .infer_output_func = op_infer_shape_unary_operator,
                                 .constraints = { OP_SETTING_CONSTRAINT_END() } };

const op_tp_t op_elu_tp = {
    .op_tp_code = OP_ELU,
    .name = "elu",
    .min_input_size = 1,
    .max_input_size = 1,
    .min_output_size = 1,
    .max_output_size = 1,
    .infer_output_func = op_infer_shape_unary_operator,
    .constraints = {OP_SETTING_CONSTRAINT_OPTIONAL(SETTING_ELU_ALPHA, dtFLOAT32, 1.0),
                    OP_SETTING_CONSTRAINT_END()}
};
