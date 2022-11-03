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

#ifndef QUANT_OP_SETTINGS_H
#define QUANT_OP_SETTINGS_H

#include "art/op_settings.h"

//#define USE_FIXED_POINT_ONLY

#define SETTING_DECLARE(group, code) ((group << 16) | (code))

#define SETTING_QUANT_IALPHA      SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0101)
#define SETTING_QUANT_IZERO_POINT SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0102)
#define SETTING_QUANT_IBITS       SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0103)
#define SETTING_QUANT_WALPHA      SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0104)
#define SETTING_QUANT_WZERO_POINT SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0105)
#define SETTING_QUANT_WBITS       SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0106)
#define SETTING_QUANT_OALPHA      SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0107)
#define SETTING_QUANT_OZERO_POINT SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0108)
#define SETTING_QUANT_OBITS       SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x0109)

/* quantization type */
#define SETTING_QUANT_QTYPE     SETTING_DECLARE(OP_GROUP_CODE_QUANT, 0x010a)
#define SETTING_QUANT_DEFAULT   0x0000
#define SETTING_QUANT_BIASED    0x0001
#define SETTING_QUANT_SYMMETRIC 0x0002

#endif // QUANT_OP_SETTINGS_H
