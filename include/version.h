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

#pragma once
#define STR_VERSION(ver)  _STR_VERSION(ver)
#define _STR_VERSION(ver) #ver

constexpr char const *nart_version_major = STR_VERSION(NART_VERSION_MAJOR);
constexpr char const *nart_version_minor = STR_VERSION(NART_VERSION_MINOR);
constexpr char const *nart_version_patch = STR_VERSION(NART_VERSION_PATCH);

#define CAT_VERSION(major, minor, patch)  _CAT_VERSION(major, minor, patch)
#define _CAT_VERSION(major, minor, patch) #major "." #minor "." #patch
constexpr const char nart_version[]
    = CAT_VERSION(NART_VERSION_MAJOR, NART_VERSION_MINOR, NART_VERSION_PATCH);
#undef CAT_VERSION
#undef _CAT_VERSION
