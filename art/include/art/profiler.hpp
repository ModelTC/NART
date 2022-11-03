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

#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <iostream>
#include <memory> // shared_ptr
#include <mutex>

class Profiler {
public:
    virtual void pointcut_begin(const char *module_tag, const char *pointcut_tag, int64_t id) = 0;
    virtual void pointcut_end(const char *module_tag, const char *pointcut_tag, int64_t id) = 0;

    Profiler() = default;
    virtual ~Profiler();
    // delete all copy&move constructors/assignment operators.
    Profiler(const Profiler &) = delete;
    Profiler(Profiler &&) = delete;
    Profiler &operator=(const Profiler &) = delete;
    Profiler &operator=(Profiler &&) = delete;
};

class ProfilerManager {
public:
    using Ptr = std::shared_ptr<ProfilerManager>;
    static Ptr getInstance();
    void subscribe(std::shared_ptr<Profiler> profiler);
    void pointcut_begin(const char *module_tag, const char *pointcut_tag, int64_t id);
    void pointcut_end(const char *module_tag, const char *pointcut_tag, int64_t id);

private:
    static Ptr instance;
    std::weak_ptr<Profiler> receiver;
    ProfilerManager();
};

#endif
