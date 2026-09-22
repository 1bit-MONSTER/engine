// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// parallel.h: a fixed pool of worker threads running one parallel_for at a time.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace onebit {

class ThreadPool {
public:
    // n_threads == 0 uses std::thread::hardware_concurrency().
    explicit ThreadPool(size_t n_threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return workers_.size() + 1; }

    // Split [0, n) into contiguous chunks and call fn(begin, end) on each,
    // using the calling thread as one of the workers. Blocks until done.
    // Not reentrant: fn must not call parallel_for on the same pool.
    void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn);

private:
    void worker_loop(size_t index);
    void run_chunk(size_t index);

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    const std::function<void(size_t, size_t)>* job_ = nullptr;
    size_t job_n_ = 0;
    size_t generation_ = 0;
    size_t pending_ = 0;
    bool stop_ = false;
};

}  // namespace onebit
