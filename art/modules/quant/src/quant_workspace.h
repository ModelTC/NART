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

#ifndef QUANT_WORKSPACE_H
#define QUANT_WORKSPACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quant_workspace_t {
    workspace_t ws;
    char *name;
} quant_workspace_t;

#define quant_WORKSPACE_STREAM(ws) ((quant_workspace_t *)(ws))->stream

#ifdef __cplusplus
}
#endif

#endif // QUANT_WORKSPACE_H
