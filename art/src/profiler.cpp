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

#include <iostream>
#include <memory> // shared_ptr
#include <mutex>

#include "art/profiler.h"
#include "art/profiler.hpp"

std::shared_ptr<ProfilerManager> ProfilerManager::instance = nullptr;

ProfilerManager::ProfilerManager() { }

std::shared_ptr<ProfilerManager> ProfilerManager::getInstance()
{
    if (instance == nullptr) {
        instance = Ptr(new ProfilerManager());
    }
    return instance;
}

void ProfilerManager::subscribe(std::shared_ptr<Profiler> profiler) { receiver = (profiler); }

void ProfilerManager::pointcut_begin(const char *module_tag, const char *pointcut_tag, int64_t id)
{
    std::shared_ptr<Profiler> shared_receiver = receiver.lock();
    if (shared_receiver != nullptr) {
        shared_receiver->pointcut_begin(module_tag, pointcut_tag, id);
    }
}

void ProfilerManager::pointcut_end(const char *module_tag, const char *pointcut_tag, int64_t id)
{
    std::shared_ptr<Profiler> shared_receiver = receiver.lock();
    if (shared_receiver != nullptr) {
        shared_receiver->pointcut_end(module_tag, pointcut_tag, id);
    }
}

Profiler::~Profiler() { }

static int profiler_case_id = 0;

void profiler_pointcut_begin(const char *module_tag, const char *pointcut_tag)
{
    auto profiler_manager = ProfilerManager::getInstance();
    profiler_manager->pointcut_begin(
        module_tag, pointcut_tag, reinterpret_cast<int64_t>(&profiler_case_id));
}

void profiler_pointcut_end(const char *module_tag, const char *pointcut_tag)
{
    auto profiler_manager = ProfilerManager::getInstance();
    profiler_manager->pointcut_end(
        module_tag, pointcut_tag, reinterpret_cast<int64_t>(&profiler_case_id));
}
