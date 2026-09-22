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
#include "parallel.h"

#include <algorithm>

namespace onebit {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0) n_threads = std::max(1u, std::thread::hardware_concurrency());
    for (size_t i = 1; i < n_threads; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    start_cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::run_chunk(size_t index) {
    const size_t n = job_n_, parts = size();
    const size_t begin = n * index / parts, end = n * (index + 1) / parts;
    if (begin < end) (*job_)(begin, end);
}

void ThreadPool::worker_loop(size_t index) {
    size_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mu_);
            start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
        }
        run_chunk(index);
        {
            std::lock_guard lock(mu_);
            if (--pending_ == 0) done_cv_.notify_one();
        }
    }
}

void ThreadPool::parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
    if (workers_.empty() || n < 2) {
        if (n > 0) fn(0, n);
        return;
    }
    {
        std::lock_guard lock(mu_);
        job_ = &fn;
        job_n_ = n;
        pending_ = workers_.size();
        ++generation_;
    }
    start_cv_.notify_all();
    run_chunk(0);
    std::unique_lock lock(mu_);
    done_cv_.wait(lock, [&] { return pending_ == 0; });
    job_ = nullptr;
}

}  // namespace onebit
