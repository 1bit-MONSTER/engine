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
//
// stream_bridge.h -- hands a streamed reply from the thread that generates it to the thread that writes it.
//
// The NPU route used to build every server-sent-event frame of a streamed reply and send them all after the
// generation had ended, so a client's time to first token was the whole request (engine#1785: 14 chunks, every
// one arriving at 9.668 s of a 9.668 s reply).  httplib writes a response body only after the handler returns,
// from a content provider it calls on the connection's thread, so frames can leave while tokens are still being
// produced only if the generation runs on another thread.  This is the piece between the two.
//
//   producer (the generation)             consumer (the connection)
//   push(frame) ... push(frame)           wait_first()   -> a frame is ready, or it ended without one
//   finish()                              next(frame)    -> the next frame in order; false when ended and drained
//   cancelled()  <- the client went away  cancel()
//
// The decision that needs wait_first(): a request that fails before its first frame (a prompt over the context,
// a malformed field) must still be answered with its own status code, and a status cannot change once a stream
// has started.  So the connection thread waits for the first frame OR the end, and only then chooses between
// "an ordinary error response" and "a stream".
//
// Frames are delivered in the order pushed, each exactly once.  finish() may be called once; push() after it is
// ignored.  Nothing here knows about HTTP.

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

namespace onebit::npu {

class StreamBridge {
public:
    // Producer: queue one frame.  Returns false when the consumer has cancelled, so the producer can stop early.
    bool push(std::string frame) {
        std::lock_guard<std::mutex> lock(mu_);
        if (finished_) return !cancelled_;
        frames_.push_back(std::move(frame));
        ++pushed_;
        cv_.notify_all();
        return !cancelled_;
    }

    // Producer: no more frames.
    void finish() {
        std::lock_guard<std::mutex> lock(mu_);
        finished_ = true;
        cv_.notify_all();
    }

    // Producer: has the consumer gone?
    bool cancelled() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cancelled_;
    }

    // Consumer: block until a frame has been pushed or the producer has finished.  True when at least one frame
    // was ever pushed (a stream should start), false when it ended with none (answer some other way).
    bool wait_first() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return pushed_ > 0 || finished_; });
        return pushed_ > 0;
    }

    // Consumer: the next frame, blocking while there is none and the producer is still going.  False when the
    // producer has finished and every frame has been taken, or after cancel().
    bool next(std::string& frame) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !frames_.empty() || finished_ || cancelled_; });
        if (cancelled_ || frames_.empty()) return false;
        frame = std::move(frames_.front());
        frames_.pop_front();
        return true;
    }

    // Consumer: stop.  The producer sees it at its next push() or cancelled().
    void cancel() {
        std::lock_guard<std::mutex> lock(mu_);
        cancelled_ = true;
        cv_.notify_all();
    }

    // Has any frame ever been pushed?  (The producer asks this to decide whether a failure can still be an ordinary
    // error response or has to be reported inside the stream.)
    bool started() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pushed_ > 0;
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(mu_);
        return finished_;
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> frames_;
    size_t pushed_ = 0;
    bool finished_ = false;
    bool cancelled_ = false;
};

}  // namespace onebit::npu
