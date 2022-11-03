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

#ifdef TIO
#define TI(tag)                           \
    cudaEvent_t _event_start_##tag;       \
    cudaEvent_t _event_end_##tag;         \
    float _event_time_##tag;              \
    cudaEventCreate(&_event_start_##tag); \
    cudaEventCreate(&_event_end_##tag);   \
    cudaEventRecord(_event_start_##tag);

#define TO(tag, str)                                                                \
    cudaEventRecord(_event_end_##tag);                                              \
    cudaEventSynchronize(_event_end_##tag);                                         \
    cudaEventElapsedTime(&_event_time_##tag, _event_start_##tag, _event_end_##tag); \
    printf("%s: %.6fms\n", str, _event_time_##tag);

#else

#define TI(tag)
#define TO(tag, str)

#endif // TIO
