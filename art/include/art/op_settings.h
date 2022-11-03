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

#define SETTING_END 0xffffffff
#define MAX_SLICE   256

#include "settings_helper.h"

/* op conv_2d & deconv_2d & conv_2d_wino & deform_conv_2d */
#define SETTING_CONV_2D_NUM_OUTPUT   1
#define SETTING_CONV_2D_PAD_H        2
#define SETTING_CONV_2D_PAD_W        3
#define SETTING_CONV_2D_KERNEL_H     4
#define SETTING_CONV_2D_KERNEL_W     5
#define SETTING_CONV_2D_STRIDE_H     6
#define SETTING_CONV_2D_STRIDE_W     7
#define SETTING_CONV_2D_GROUP        8
#define SETTING_CONV_2D_RELU_FLAG    9
#define SETTING_CONV_2D_DEFORM_GROUP 10
#define SETTING_CONV_2D_HOLE_H       11
#define SETTING_CONV_2D_HOLE_W       12
/* op conv_nd & deconv_nd & conv_nd_wino & deform_conv_nd */
#define SETTING_CONV_ND_NUM_OUTPUT   1
#define SETTING_CONV_ND_PAD          2
#define SETTING_CONV_ND_KERNEL       3
#define SETTING_CONV_ND_STRIDE       4
#define SETTING_CONV_ND_GROUP        5
#define SETTING_CONV_ND_RELU_FLAG    6
#define SETTING_CONV_ND_DEFORM_GROUP 7
#define SETTING_CONV_ND_HOLE         8
/* op matmul */
#define SETTING_MATMUL_RELU_FLAG 1
/* op lrn */
#define SETTING_LRN_LOCAL_SIZE  1
#define SETTING_LRN_ALPHA       2
#define SETTING_LRN_BETA        3
#define SETTING_LRN_K           4
#define SETTING_LRN_NORM_REGION 5
/* op pool */
#define SETTING_POOL_METHOD    1
#define SETTING_POOL_PAD_H     2
#define SETTING_POOL_PAD_W     3
#define SETTING_POOL_KERNEL_H  4
#define SETTING_POOL_KERNEL_W  5
#define SETTING_POOL_STRIDE_H  6
#define SETTING_POOL_STRIDE_W  7
#define SETTING_POOL_CEIL_MODE 8
/* pool_method */
#define SETTING_POOL_MAX 0
#define SETTING_POOL_AVE 1
/* op ip */
#define SETTING_IP_NUM_OUTPUT 1
#define SETTING_IP_RELU_FLAG  2
#define SETTING_IP_AXIS       3
/* op bn */
#define SETTING_BN_EPS 1
/* op batchnorm */
#define SETTING_BATCHNORM_EPS 1

/* op concat */
#define SETTING_CONCAT_AXIS 1
/* op slice */
#define SETTING_SLICE_AXIS  1
#define SETTING_SLICE_POINT 2
/* op interp */
#define SETTING_INTERP_HEIGHT        1
#define SETTING_INTERP_WIDTH         2
#define SETTING_INTERP_ZOOM_FACTOR   3
#define SETTING_INTERP_SHRINK_FACTOR 4
#define SETTING_INTERP_PAD_BEG       5
#define SETTING_INTERP_PAD_END       6
#define SETTING_INTERP_TYPE          7
#define SETTING_INTERP_NEAREST       0
#define SETTING_INTERP_BILINEAR      1
/* op softmax */
#define SETTING_SOFTMAX_AXIS 1

/* op eltwise */
#define SETTING_ELTWISE_OPERATION 1
#define SETTING_ELTWISE_COEFF     2
#define SETTING_ELTWISE_RELU_FLAG 3
#define SETTING_ELTWISE_OP_PROD   0
#define SETTING_ELTWISE_OP_SUM    1
#define SETTING_ELTWISE_OP_MAX    2

/* op scale */
#define SETTING_SCALE_BIAS_TERM 1

/* op reshape */
#define SETTING_RESHAPE_DIMS     1
#define SETTING_RESHAPE_DIM_SIZE 2
#define SETTING_RESHAPE_AXIS     3
#define SETTING_RESHAPE_NUM_AXES 4

/* op subpixel */
#define SETTING_SUBPIXEL_METHOD 1
#define SETTING_SUBPIXEL_SAMPLE 2

/* op prelu */
#define SETTING_PRELU_SHARE 1

/* op roipooling */
#define SETTING_ROIPOOLING_SPATIAL_SCALE 1
#define SETTING_ROIPOOLING_POOLED_HEIGHT 2
#define SETTING_ROIPOOLING_POOLED_WIDTH  3

/* op psroipooling */
#define SETTING_PSROIPOOLING_SPATIAL_SCALE 1
#define SETTING_PSROIPOOLING_OUTPUT_DIM    2
#define SETTING_PSROIPOOLING_GROUP_SIZE    3
#define SETTING_PSROIPOOLING_SAMPLE_NUM    4

/* op roialignpooling */
#define SETTING_ROIALIGNPOOLING_SPATIAL_SCALE 1
#define SETTING_ROIALIGNPOOLING_POOLED_HEIGHT 2
#define SETTING_ROIALIGNPOOLING_POOLED_WIDTH  3
#define SETTING_ROIALIGNPOOLING_SAMPLE_NUM    4

/* op podroialignpooling */
#define SETTING_PODROIALIGNPOOLING_SPATIAL_SCALE 1
#define SETTING_PODROIALIGNPOOLING_POOLED_HEIGHT 2
#define SETTING_PODROIALIGNPOOLING_POOLED_WIDTH  3
#define SETTING_PODROIALIGNPOOLING_SAMPLE_NUM    4

/* transpose */
#define SETTING_TRANSPOSE_DIMS    1
#define SETTING_TRANSPOSE_EXGAXIS 2

/* heatmap2coord */
#define SETTING_HEATMAP2COORD_COORD_H    1
#define SETTING_HEATMAP2COORD_COORD_W    2
#define SETTING_HEATMAP2COORD_REPOSITION 3

/* op psroimaskpooling */
#define SETTING_PSROIMASKPOOLING_SPATIAL_SCALE 1
#define SETTING_PSROIMASKPOOLING_ROI_SCALE     2
#define SETTING_PSROIMASKPOOLING_BIN_SCALE     3
#define SETTING_PSROIMASKPOOLING_OUTPUT_DIM    4
#define SETTING_PSROIMASKPOOLING_GROUP_SIZE    5
#define SETTING_PSROIMASKPOOLING_SAMPLE_NUM    6

/* bilateralslice */
#define SETTING_BILATERALSLICE_COE    1
#define SETTING_BILATERALSLICE_OFFSET 2

/* op shufflechannel */
#define SETTING_SHUFFLECHANNEL_GROUP 1

/* op psroialignpooling */
#define SETTING_PSROIALIGNPOOLING_SPATIAL_SCALE 1
#define SETTING_PSROIALIGNPOOLING_OUTPUT_DIM    2
#define SETTING_PSROIALIGNPOOLING_GROUP_SIZE    3
#define SETTING_PSROIALIGNPOOLING_SAMPLE_NUM    4

/* op correlation */
#define SETTING_CORRELATION_GROUPS 1

/*op pad*/
#define SETTING_PAD_MODE  1
#define SETTING_PAD_VALUE 2
#define SETTING_PAD_PADS  3

/* op quant_dequant */
#define SETTING_QUANT_DEQUANT_QMIN  1
#define SETTING_QUANT_DEQUANT_QMAX  2
#define SETTING_QUANT_DEQUANT_SCALE 3

/*op instancenorm*/
#define SETTING_INSTANCENORM_EPS 1

/*op reduce*/
#define SETTING_REDUCE_AXES     1
#define SETTING_REDUCE_KEEPDIMS 2

/*op correlation1d */
#define SETTING_CORRELATION1D_MAX_DISPLACEMENT 1
#define SETTING_CORRELATION1D_KERNEL_SIZE      2
#define SETTING_CORRELATION1D_SINGLE_DIRECTION 3
#define SETTING_CORRELATION1D_PAD              4

/*op lpnormalization */
#define SETTING_LPNORMALIZATION_P    1
#define SETTING_LPNORMALIZATION_AXIS 2

/*op gather */
#define SETTING_GATHER_AXIS 1

/*op argmax */
#define SETTING_ARGMAX_AXIS              1
#define SETTING_ARGMAX_KEEPDIMS          2
#define SETTING_ARGMAX_SELECT_LAST_INDEX 3

/*op gridsample*/
#define SETTING_GRIDSAMPLE_MODE          1
#define SETTING_GRIDSAMPLE_PADDING_MODE  2
#define SETTING_GRIDSAMPLE_ALIGN_CORNERS 3

// gridsample mode
#define GRIDSAMPLE_MODE_BILINEAR 1
#define GRIDSAMPLE_MODE_NEREAST  2

// gridsample padding
#define GRIDSAMPLE_PADDING_ZEROS      1
#define GRIDSAMPLE_PADDING_BORDER     2
#define GRIDSAMPLE_PADDING_REFLECTION 3

/*op unfold (aka im2col)*/
#define SETTING_UNFOLD_PAD_H    1
#define SETTING_UNFOLD_PAD_W    2
#define SETTING_UNFOLD_KERNEL_H 3
#define SETTING_UNFOLD_KERNEL_W 4
#define SETTING_UNFOLD_STRIDE_H 5
#define SETTING_UNFOLD_STRIDE_W 6
#define SETTING_UNFOLD_HOLE_H   7
#define SETTING_UNFOLD_HOLE_W   8

/* op topk */
#define SETTING_TOPK_AXIS 1
#define SETTING_TOPK_K    2

/*op lstm */
#define SETTING_LSTM_HIDDEN_SIZE        1
#define SETTING_LSTM_DIRECTION          2
#define SETTING_LSTM_INPUT_FORGET       3
#define SETTING_LSTM_CLIP               4
#define SETTING_LSTM_CLIP_EXIST         5
#define SETTING_LSTM_ACTIVATION_F       6
#define SETTING_LSTM_ACTIVATION_G       7
#define SETTING_LSTM_ACTIVATION_H       8
#define SETTING_LSTM_ACTIVATION_ALPHA_F 9
#define SETTING_LSTM_ACTIVATION_ALPHA_G 10
#define SETTING_LSTM_ACTIVATION_ALPHA_H 11
#define SETTING_LSTM_ACTIVATION_BETA_F  12
#define SETTING_LSTM_ACTIVATION_BETA_G  13
#define SETTING_LSTM_ACTIVATION_BETA_H  14
#define SETTING_LSTM_OUTPUT_SIZE        15

// lstm direction mode
#define LSTM_DIRECTION_FORWARD       1
#define LSTM_DIRECTION_REVERSE       2
#define LSTM_DIRECTION_BIDIRECTIONAL 3

// lstm activation function mode
#define LSTM_ACTIVATION_RELU    1
#define LSTM_ACTIVATION_TANH    2
#define LSTM_ACTIVATION_SIGMOID 3

/* op hardsigmoid */
#define SETTING_HARDSIGMOID_ALPHA 1
#define SETTING_HARDSIGMOID_BETA  2

/* op cast */
#define SETTING_CAST_DTYPE 1

/* op scatternd */
#define SETTING_SCATTERND_REDUCTION 1

/* op Elu, the activation */
#define SETTING_ELU_ALPHA 1
