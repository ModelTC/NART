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

#include <math.h> // used in implementation of activation fuction: exp(..)

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../utils/sgemm.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t hidden_size;
    uint32_t direction;
    uint32_t input_forget;
    // TODO: clip and clip_exit get but not implemented yet
    float clip;
    uint32_t clip_exist; // if clip_exist == 1, we apply clip to results
    float activation_alpha_f, activation_alpha_g, activation_alpha_h;
    float activation_beta_f, activation_beta_g, activation_beta_h;
    uint32_t output_size;
    uint32_t num_directions; // related to op->direction, forward and reverse have num_directions 1
    // and bidirectional has 2
    uint32_t input_size;
    uint32_t seq_length;
    uint32_t batch_size;
    void (*activation_f)(const float *, float *, int, float, float);
    void (*activation_g)(const float *, float *, int, float, float);
    void (*activation_h)(const float *, float *, int, float, float);
    mem_t *it;
    mem_t *ft;
    mem_t *ct;
    mem_t *ot;
    mem_t *cell;
    mem_t *hidden;
    mem_t *activated_cell;
} op_lstm_t;
op_lstm_t *op_default_lstm_tp_alloc(workspace_t *ws);
void op_default_lstm_tp_config(op_t *op);
void op_default_lstm_tp_destroy(op_t *op);
void op_default_lstm_tp_dealloc(op_t *op);
void op_default_lstm_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
// the implementation of activation fuctions
static void lstm_sigmoid(const float *X, float *Y, int length, float alpha, float beta)
{
    /* X: Input, Pointer of head of array
    Y: Output, Pinter of head of array
    length: how many data wo need to apply activation
    alpha, beta: parameter of activation functions. Some activation do not have alpha and beta
    */
    for (int i = 0; i < length; i++) {
        Y[i] = 1 / (1 + exp(-X[i]));
    }
}

static void lstm_tanh(const float *X, float *Y, int length, float alpha, float beta)
{
    for (int i = 0; i < length; i++) {
        float s1 = exp(X[i]) - exp(-X[i]);
        float s2 = exp(X[i]) + exp(-X[i]);
        Y[i] = s1 / s2;
    }
}

static void lstm_relu(const float *X, float *Y, int length, float alpha, float beta)
{
    for (int i = 0; i < length; i++) {
        Y[i] = X[i] > 0 ? X[i] : 0;
    }
}

op_lstm_t *op_default_lstm_tp_alloc(workspace_t *ws)
{
    op_lstm_t *res = (op_lstm_t *)malloc(sizeof(op_lstm_t));
    memset(res, 0, sizeof(op_lstm_t));
    return res;
}

void op_default_lstm_tp_config(op_t *op)
{
    // Get parameters of LSTM
    op_lstm_t *lstm_op = (op_lstm_t *)op;
    CHECK(op_setting_single_get(op, SETTING_LSTM_HIDDEN_SIZE, dtINT32, &lstm_op->hidden_size));
    CHECK(op_setting_single_get(op, SETTING_LSTM_DIRECTION, dtINT32, &lstm_op->direction));
    CHECK(op_setting_single_get(op, SETTING_LSTM_INPUT_FORGET, dtINT32, &lstm_op->input_forget));
    CHECK(op_setting_single_get(op, SETTING_LSTM_CLIP, dtFLOAT32, &lstm_op->clip));
    CHECK(op_setting_single_get(op, SETTING_LSTM_CLIP_EXIST, dtINT32, &lstm_op->clip_exist));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_ALPHA_F, dtFLOAT32, &lstm_op->activation_alpha_f));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_ALPHA_G, dtFLOAT32, &lstm_op->activation_alpha_g));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_ALPHA_H, dtFLOAT32, &lstm_op->activation_alpha_h));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_BETA_F, dtFLOAT32, &lstm_op->activation_beta_f));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_BETA_G, dtFLOAT32, &lstm_op->activation_beta_g));
    CHECK(op_setting_single_get(
        op, SETTING_LSTM_ACTIVATION_BETA_H, dtFLOAT32, &lstm_op->activation_beta_h));
    CHECK(op_setting_single_get(op, SETTING_LSTM_OUTPUT_SIZE, dtINT32, &lstm_op->output_size));

    int activation_mode_f, activation_mode_g, activation_mode_h;
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_F, dtINT32, &activation_mode_f));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_G, dtINT32, &activation_mode_g));
    CHECK(op_setting_single_get(op, SETTING_LSTM_ACTIVATION_H, dtINT32, &activation_mode_h));

    // Set activation functions to func_ptr according to activation_mode
    // See the setting of mode in op_settings.h
    switch (activation_mode_f) {
    case LSTM_ACTIVATION_RELU:
        lstm_op->activation_f = &lstm_relu;
        break;

    case LSTM_ACTIVATION_TANH:
        lstm_op->activation_f = &lstm_tanh;
        break;

    case LSTM_ACTIVATION_SIGMOID:
        lstm_op->activation_f = &lstm_sigmoid;
        break;

    default:
        CHECK(false);
        break;
    }

    switch (activation_mode_g) {
    case LSTM_ACTIVATION_RELU:
        lstm_op->activation_g = &lstm_relu;
        break;

    case LSTM_ACTIVATION_TANH:
        lstm_op->activation_g = &lstm_tanh;
        break;

    case LSTM_ACTIVATION_SIGMOID:
        lstm_op->activation_g = &lstm_sigmoid;
        break;

    default:
        CHECK(false);
        break;
    }

    switch (activation_mode_h) {
    case LSTM_ACTIVATION_RELU:
        lstm_op->activation_h = &lstm_relu;
        break;

    case LSTM_ACTIVATION_TANH:
        lstm_op->activation_h = &lstm_tanh;
        break;

    case LSTM_ACTIVATION_SIGMOID:
        lstm_op->activation_h = &lstm_sigmoid;
        break;

    default:
        CHECK(false);
        break;
    }

    // forward and reverse have num_directions 1, bidirectional has 2
    if (lstm_op->direction == LSTM_DIRECTION_BIDIRECTIONAL)
        lstm_op->num_directions = 2;
    else
        lstm_op->num_directions = 1;
}

void op_default_lstm_tp_destroy(op_t *op)
{
    mem_delete(((op_lstm_t *)op)->it);
    mem_delete(((op_lstm_t *)op)->ft);
    mem_delete(((op_lstm_t *)op)->ct);
    mem_delete(((op_lstm_t *)op)->ot);
    mem_delete(((op_lstm_t *)op)->cell);
    mem_delete(((op_lstm_t *)op)->hidden);
    mem_delete(((op_lstm_t *)op)->activated_cell);
}

void op_default_lstm_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

typedef void (*activation_ptr)(const float *, float *, int, float, float);

void lstm_gate(
    float *output, activation_ptr activation, const float alpha, const float beta, const float *X,
    const float *hidden, const float *W, const float *R, const float *Bw, const float *Br,
    const int batch_size, const int hidden_size, const int input_size)
{
    /* An overview formula of lstm_gate: output = activation(X*(W^T) + hidden*(R^T) +  Bw + Br)
     */
    memset(output, 0, batch_size * hidden_size * sizeof(float));
    sgemm_AxB(batch_size, hidden_size, input_size, X, W, output); // X*(W^T)
    sgemm_AxB(batch_size, hidden_size, hidden_size, hidden, R, output); // hidden*(R^T)
    // Bw + Br
    float *ite = output;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            *ite += Bw[j] + Br[j];
            ite++;
        }
    }
    (*activation)(output, output, batch_size * hidden_size, alpha, beta); // activation(...)
}

static void op_default_lstm_run(op_t *op)
{
    /*
    Overview Equations of LSTM
    it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Bwi + Bri)
    ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Bwf + Brf)
    ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Bwc + Brc)
    Ct = ft (.) Ct-1 + it (.) ct
    ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Bwo + Bro)
    Ht = ot (.) h(Ct)
    */
    // Inputs and Outputs
    const float *X = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    const float *W = (const float *)mem_cpu_data(op->input_tensors[1]->mem);
    const float *R = (const float *)mem_cpu_data(op->input_tensors[2]->mem);
    const float *B = (const float *)mem_cpu_data(op->input_tensors[3]->mem);
    const float *seq_len = (const float *)mem_cpu_data(op->input_tensors[4]->mem);
    float *Y = (float *)mem_cpu_data(op->output_tensors[0].mem);

    // Parameters and temp spaces
    op_lstm_t *lstm_op = (op_lstm_t *)op;
    float *it = (float *)mem_cpu_data(lstm_op->it);
    float *ft = (float *)mem_cpu_data(lstm_op->ft);
    float *ct = (float *)mem_cpu_data(lstm_op->ct);
    float *ot = (float *)mem_cpu_data(lstm_op->ot);
    float *cell = (float *)mem_cpu_data(lstm_op->cell);
    float *activated_cell = (float *)mem_cpu_data(lstm_op->activated_cell);
    float *hidden = (float *)mem_cpu_data(lstm_op->hidden);
    float *hidden_prev;
    float *hidden_output;
    const float *Xs;
    const float *Wi, *Wo, *Wf, *Wc;
    const float *Ri, *Ro, *Rf, *Rc;
    const float *Bwi, *Bwo, *Bwf, *Bwc;
    const float *Bri, *Bro, *Brf, *Brc;

    // Alias
    const int sl = lstm_op->seq_length;
    const int bs = lstm_op->batch_size;
    const int hs = lstm_op->hidden_size;
    const int is = lstm_op->input_size;
    const int nd = lstm_op->num_directions;
    const float alpha_f = lstm_op->activation_alpha_f;
    const float alpha_g = lstm_op->activation_alpha_g;
    const float alpha_h = lstm_op->activation_alpha_h;
    const float beta_f = lstm_op->activation_beta_f;
    const float beta_g = lstm_op->activation_beta_g;
    const float beta_h = lstm_op->activation_beta_h;
    const float deviation = 1e-2;

    // The LSTM forward phase
    if (lstm_op->direction == LSTM_DIRECTION_FORWARD
        || lstm_op->direction == LSTM_DIRECTION_BIDIRECTIONAL) {

        // init the cell and hidden
        // TODO: get the init hidden and cell from input[5] and input[6]
        memset(cell, 0, bs * hs * sizeof(float));
        memset(hidden, 0, bs * hs * sizeof(float));

        const int direction = 0;

        // Calculate the head pointer of each array
        Wi = &W[direction * 4 * hs * is];
        Wo = &W[direction * 4 * hs * is + 1 * hs * is];
        Wf = &W[direction * 4 * hs * is + 2 * hs * is];
        Wc = &W[direction * 4 * hs * is + 3 * hs * is];
        Ri = &R[direction * 4 * hs * hs];
        Ro = &R[direction * 4 * hs * hs + 1 * hs * hs];
        Rf = &R[direction * 4 * hs * hs + 2 * hs * hs];
        Rc = &R[direction * 4 * hs * hs + 3 * hs * hs];
        Bwi = &B[direction * 8 * hs];
        Bwo = &B[direction * 8 * hs + 1 * hs];
        Bwf = &B[direction * 8 * hs + 2 * hs];
        Bwc = &B[direction * 8 * hs + 3 * hs];
        Bri = &B[direction * 8 * hs + 4 * hs];
        Bro = &B[direction * 8 * hs + 5 * hs];
        Brf = &B[direction * 8 * hs + 6 * hs];
        Brc = &B[direction * 8 * hs + 7 * hs];

        // for iteration of sequence length dimension
        for (int seq_index = 0; seq_index < sl; seq_index++) {

            hidden_prev = seq_index == 0 ? hidden
                                         : &Y[(seq_index - 1) * nd * bs * hs + direction * bs * hs];
            hidden_output = &Y[seq_index * nd * bs * hs + direction * bs * hs];
            Xs = &X[seq_index * bs * is];

            /*
            it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Bwi + Bri)
            ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Bwf + Brf)
            ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Bwc + Brc)
            ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Bwo + Bro)
            */
            lstm_gate(
                it, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wi, Ri, Bwi, Bri, bs,
                hs, is);
            lstm_gate(
                ft, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wf, Rf, Bwf, Brf, bs,
                hs, is);
            lstm_gate(
                ct, lstm_op->activation_g, alpha_h, beta_g, Xs, hidden_prev, Wc, Rc, Bwc, Brc, bs,
                hs, is);
            lstm_gate(
                ot, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wo, Ro, Bwo, Bro, bs,
                hs, is);

            // Ct = ft (.) Ct-1 + it (.) ct
            for (int i = 0; i < bs; i++) {
                if (seq_index < seq_len[i] - deviation)
                    for (int j = 0; j < hs; j++) {
                        int index = i * hs + j;
                        cell[index] = cell[index] * ft[index] + it[index] * ct[index];
                    }
            }

            (*lstm_op->activation_h)(cell, activated_cell, bs * hs, alpha_h, beta_h);

            // Ht = ot (.) h(Ct)
            for (int i = 0; i < bs; i++) {
                if (seq_index < seq_len[i] - deviation)
                    for (int j = 0; j < hs; j++) {
                        int index = i * hs + j;
                        hidden_output[index] = ot[index] * activated_cell[index];
                    }
                else
                    // if exceed the seq_len, just copy the last effective hidden
                    memcpy(&hidden_output[i * hs], &hidden_prev[i * hs], sizeof(float) * hs);
            }
        }

        // Copy the last hidden to Y_h (output[1])
        if (op->output_size >= 2) {
            float *Y_h = (float *)mem_cpu_data(op->output_tensors[1].mem);
            float *Y_h_di = &Y_h[direction * bs * hs];
            memcpy(Y_h_di, hidden_output, sizeof(float) * bs * hs);
        }

        // Copy the last cell to Y_c (output[2])
        if (op->output_size >= 3) {
            float *Y_c = (float *)mem_cpu_data(op->output_tensors[2].mem);
            float *Y_c_di = &Y_c[direction * bs * hs];
            memcpy(Y_c_di, cell, sizeof(float) * bs * hs);
        }
    } // end of if FORWARD or BIDIRECTIONAL

    // The LSTM reverse phase
    if (lstm_op->direction == LSTM_DIRECTION_REVERSE
        || lstm_op->direction == LSTM_DIRECTION_BIDIRECTIONAL) {

        // init the cell and hidden
        // TODO: get the init hidden and cell from input[5] and input[6]
        memset(cell, 0, bs * hs * sizeof(float));
        memset(hidden, 0, bs * hs * sizeof(float));

        int direction;

        if (lstm_op->direction == LSTM_DIRECTION_BIDIRECTIONAL)
            direction = 1;
        else if (lstm_op->direction == LSTM_DIRECTION_REVERSE)
            direction = 0;

        // Calculate the head pointer of each array
        Wi = &W[direction * 4 * hs * is];
        Wo = &W[direction * 4 * hs * is + 1 * hs * is];
        Wf = &W[direction * 4 * hs * is + 2 * hs * is];
        Wc = &W[direction * 4 * hs * is + 3 * hs * is];
        Ri = &R[direction * 4 * hs * hs];
        Ro = &R[direction * 4 * hs * hs + 1 * hs * hs];
        Rf = &R[direction * 4 * hs * hs + 2 * hs * hs];
        Rc = &R[direction * 4 * hs * hs + 3 * hs * hs];
        Bwi = &B[direction * 8 * hs];
        Bwo = &B[direction * 8 * hs + 1 * hs];
        Bwf = &B[direction * 8 * hs + 2 * hs];
        Bwc = &B[direction * 8 * hs + 3 * hs];
        Bri = &B[direction * 8 * hs + 4 * hs];
        Bro = &B[direction * 8 * hs + 5 * hs];
        Brf = &B[direction * 8 * hs + 6 * hs];
        Brc = &B[direction * 8 * hs + 7 * hs];

        for (int seq_index = sl - 1; seq_index >= 0; seq_index--) {

            hidden_prev = seq_index == sl - 1
                ? hidden
                : &Y[(seq_index + 1) * nd * bs * hs + direction * bs * hs];
            hidden_output = &Y[seq_index * nd * bs * hs + direction * bs * hs];
            Xs = &X[seq_index * bs * is];

            /*
            it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Bwi + Bri)
            ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Bwf + Brf)
            ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Bwc + Brc)
            ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Bwo + Bro)
            */
            lstm_gate(
                it, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wi, Ri, Bwi, Bri, bs,
                hs, is);
            lstm_gate(
                ft, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wf, Rf, Bwf, Brf, bs,
                hs, is);
            lstm_gate(
                ct, lstm_op->activation_g, alpha_h, beta_g, Xs, hidden_prev, Wc, Rc, Bwc, Brc, bs,
                hs, is);
            lstm_gate(
                ot, lstm_op->activation_f, alpha_f, beta_f, Xs, hidden_prev, Wo, Ro, Bwo, Bro, bs,
                hs, is);

            // Ct = ft (.) Ct-1 + it (.) ct
            for (int i = 0; i < bs; i++) {
                if (seq_index < seq_len[i] - deviation)
                    for (int j = 0; j < hs; j++) {
                        int index = i * hs + j;
                        cell[index] = cell[index] * ft[index] + it[index] * ct[index];
                    }
                // if have not touch the  seq_len, just set 0 to cell (cell is 0, so we do nothing)
            }

            (*lstm_op->activation_h)(cell, activated_cell, bs * hs, alpha_h, beta_h);

            // Ht = ot (.) h(Ct)
            for (int i = 0; i < bs; i++) {
                if (seq_index < seq_len[i] - deviation)
                    for (int j = 0; j < hs; j++) {
                        int index = i * hs + j;
                        hidden_output[index] = ot[index] * activated_cell[index];
                    }
                else
                    // if have not touch the  seq_len, just set 0 to hidden
                    memset(&hidden_output[i * hs], 0, sizeof(float) * hs);
            }
        }

        // Copy the last hidden to Y_h (output[1])
        if (op->output_size >= 2) {
            float *Y_h = (float *)mem_cpu_data(op->output_tensors[1].mem);
            float *Y_h_di = &Y_h[direction * bs * hs];
            memcpy(Y_h_di, hidden_output, sizeof(float) * bs * hs);
        }

        // Copy the last cell to Y_c (output[2])
        if (op->output_size >= 3) {
            float *Y_c = (float *)mem_cpu_data(op->output_tensors[2].mem);
            float *Y_c_di = &Y_c[direction * bs * hs];
            memcpy(Y_c_di, cell, sizeof(float) * bs * hs);
        }
    } // end of if REVERSE or BIDIRECTIONAL
}

void op_default_lstm_tp_prepare(op_t *op)
{
    int i;
    op_lstm_t *lstm_op = (op_lstm_t *)op;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_lstm_run;
        break;
    default:
        CHECK(false);
        break;
    }

    /* Overview of shapes:
    X: [seq_length, batch_size, input_size]
    W: [num_directions, 4*hidden_size, input_size]
    R: [num_directions, 4*hidden_size, hidden_size]
    B: [num_directions, 8*hidden_size]
    seq_len: [batch_size]
    Y: [seq_length, num_directions, batch_size, hidden_size]
    Y_h: [num_directions, batch_size, hidden_size]
    Y_c: [num_directions, batch_size, hidden_size
    it: [batch_size][hidden_size]
    ft: [batch_size][hidden_size]
    ct: [batch_size][hidden_size]
    ot: [batch_size][hidden_size]
    cell: [batch_size][hidden_size]
    hidden: [batch_size][hidden_size]
    */
    lstm_op->seq_length = op->input_tensors[0]->shape.dim[0];
    lstm_op->batch_size = op->input_tensors[0]->shape.dim[1];
    lstm_op->input_size = op->input_tensors[0]->shape.dim[2];

    if (NULL == lstm_op->it)
        lstm_op->it = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->it, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->ft)
        lstm_op->ft = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->ft, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->ct)
        lstm_op->ct = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->ct, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->ot)
        lstm_op->ot = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->ot, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->cell)
        lstm_op->cell = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->cell, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->activated_cell)
        lstm_op->activated_cell = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->activated_cell, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);
    if (NULL == lstm_op->hidden)
        lstm_op->hidden = mem_new(cpu_mem_tp);
    mem_alloc(lstm_op->hidden, sizeof(float) * lstm_op->batch_size * lstm_op->hidden_size);

    shape_t input_shape_0 = op->input_tensors[0]->shape;
    lstm_op->seq_length = input_shape_0.dim[0];
    lstm_op->batch_size = input_shape_0.dim[1];
    lstm_op->input_size = input_shape_0.dim[2];
}
