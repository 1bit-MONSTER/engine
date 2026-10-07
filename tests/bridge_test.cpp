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
// The device bridge (app/bridge.{h,cpp}, docs/bridge.md): the replay/resync rules, the
// k and context clamps, the error path, and the loopback endpoint, all against a
// scripted drafter, so this runs in CI with no GPU and no NPU.
#include "bridge.h"

#include <httplib.h>  // the test is the endpoint's client

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using onebit::bridge::DraftModel;
using onebit::bridge::Drafter;
using onebit::bridge::Server;

namespace {

int fails = 0;

void check(const char* name, bool ok, const std::string& detail = "") {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail.empty() ? "" : ": ", detail.c_str());
    fails += !ok;
}

std::string show(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

// A deterministic draft model whose answer depends on the whole context, not just the
// last token: if the bridge fails to replay the right tokens before drafting, the answer
// changes and the test sees it. `step` also mirrors the lane: a row written at ctx is
// overwritten, and the positions it is given are checked.
class FakeDraft : public DraftModel {
public:
    FakeDraft(int vocab, int max_ctx) : vocab_(vocab), max_ctx_(max_ctx) {}

    void reset() override {
        ++resets;
        ctx_.clear();
    }
    void step(int token, int ctx) override {
        // A row may be overwritten (a rejected draft the replay replaces), but the model
        // must never be asked to skip past the next position.
        if (ctx > int(ctx_.size()) + 1) positions_wrong = true;
        if (ctx <= int(ctx_.size())) ctx_.resize(ctx - 1);  // overwrite: what a KV row write does
        ctx_.push_back(token);
    }
    int argmax() override {
        unsigned h = 2166136261u;
        for (int t : ctx_) h = (h ^ unsigned(t)) * 16777619u;
        return int(h % unsigned(vocab_));
    }
    int max_context() const override { return max_ctx_; }
    int n_vocab() const override { return vocab_; }

    const std::vector<int>& ctx() const { return ctx_; }
    int resets = 0;
    bool positions_wrong = false;

private:
    std::vector<int> ctx_;
    int vocab_, max_ctx_;
};

// A model that fails, to check that a broken drafter is an empty draft and not a crash.
class ThrowingDraft : public DraftModel {
public:
    void reset() override {}
    void step(int, int) override { throw std::runtime_error("no NPU"); }
    int argmax() override { return 0; }
    int max_context() const override { return 1024; }
    int n_vocab() const override { return 32; }
};

std::unique_ptr<DraftModel> fake(FakeDraft** out, int vocab = 4096, int max_ctx = 4096) {
    auto* m = new FakeDraft(vocab, max_ctx);
    if (out) *out = m;
    return std::unique_ptr<DraftModel>(m);
}

std::filesystem::path write_script(const std::string& body) {
    const auto p = std::filesystem::temp_directory_path() / "bridge_test_script.txt";
    std::ofstream(p) << body;
    return p;
}

void test_script() {
    const auto path = write_script("# the bridge's test seam\n11\n\n12   # a comment\n13\n");
    std::string err;
    auto m = onebit::bridge::script_draft(path.string(), 0, 0, err);
    check("script parses", m != nullptr, err);
    if (!m) return;
    check("script answers in order", m->argmax() == 11 && m->argmax() == 12 && m->argmax() == 13);
    check("an exhausted script has no answer", m->argmax() == -1);

    std::ofstream(path) << "1\npotato\n";
    check("a bad script line is refused", onebit::bridge::script_draft(path.string(), 0, 0, err) == nullptr);
    check("a missing script is refused", onebit::bridge::script_draft(path.string() + ".nope", 0, 0, err) == nullptr);
    std::filesystem::remove(path);
}

void test_draft_and_resync() {
    FakeDraft* f = nullptr;
    Drafter d(fake(&f, 64, 4096), 3);
    std::string note;

    // Round 1: the committed sequence is prompt + [id_last], and the drafts are greedy.
    std::vector<int> a = d.draft({1, 2, 3}, 4, 3, note);
    check("round 1 drafts k", a.size() == 3 && note.empty(), show(a));
    const std::vector<int> seq1 = {1, 2, 3, 4};
    std::vector<int> want = seq1;
    want.insert(want.end(), a.begin(), a.end());
    check("the model saw the committed sequence", f->ctx() == want, show(f->ctx()));
    check("the model was never asked to skip a position", !f->positions_wrong);
    check("round 1 did not rebuild the drafter", f->resets == 0);

    // Round 2: the target accepted two drafts and emitted its own token. That sequence
    // extends round 1's, so the bridge replays only the tail: no rebuild, and the
    // rejected third draft is overwritten by the replay.
    std::vector<int> prompt2 = {1, 2, 3, 4, a[0], a[1]};
    const int target_token = 5;
    std::vector<int> b = d.draft(prompt2, target_token, 3, note);
    check("round 2 drafts k", b.size() == 3 && note.empty(), show(b));
    want = prompt2;
    want.push_back(target_token);
    want.insert(want.end(), b.begin(), b.end());
    check("round 2 replayed the tail only", f->ctx() == want, show(f->ctx()));
    check("round 2 did not rebuild the drafter", f->resets == 0);
    check("round 2 counted the replay", d.replayed() == 7);
    std::vector<int> committed = prompt2;
    committed.push_back(target_token);
    check("the drafter's committed sequence is round 2's", d.committed() == committed, show(d.committed()));

    // A sequence that does not extend what the drafter holds (a new conversation, a
    // reused cache) is replayed from the longest common prefix, not rebuilt: the model
    // is asked for numbers it has, and the rows past the new sequence are never read.
    std::vector<int> c = d.draft({9, 9}, 8, 3, note);
    check("a divergent sequence replays from the common prefix", f->resets == 0 && c.size() == 3, show(c));
    check("the rebuilt context is the new sequence", f->ctx().size() == 6, show(f->ctx()));
    check("the rebuild had no position gap", !f->positions_wrong);
}

void test_clamps_and_errors() {
    FakeDraft* f = nullptr;
    Drafter d(fake(&f, 64, 4096), 4);
    std::string note;

    check("n_max 0 means the cap", d.draft({1}, 2, 0, note).size() == 4);
    check("n_max is clamped by the cap", d.draft({1}, 2, 99, note).size() == 4);
    check("one draft is asked for", d.draft({1}, 2, 1, note).size() == 1);

    // The drafter's context bounds the round: no room means no answer, once.
    {
        FakeDraft* g = nullptr;
        Drafter small(fake(&g, 64, 6), 4);
        std::string n;
        check("a full context answers nothing", small.draft({1, 2, 3, 4, 5}, 6, 4, n).empty());
        check("and says why", n.find("full") != std::string::npos, n);
        check("no draft model call was made", g->resets == 1 && g->ctx().empty());
        check("it is not recomputed every round", small.draft({1, 2, 3, 4, 5}, 6, 4, n).empty() && n.empty() &&
                                                     g->resets == 1);
        // Room for two only.
        FakeDraft* h = nullptr;
        Drafter tight(fake(&h, 64, 8), 4);
        std::string m;
        check("the round is clamped to the room left", tight.draft({1, 2, 3, 4}, 5, 4, m).size() == 3);
    }

    // A token the drafter cannot represent (the two paths disagree on the vocabulary).
    {
        FakeDraft* g = nullptr;
        Drafter v(fake(&g, 32, 4096), 4);
        std::string n;
        check("a token outside the drafter's vocabulary is refused", v.draft({1}, 999, 4, n).empty());
        check("and says why", n.find("vocabulary") != std::string::npos, n);
    }

    // A drafter that throws is an empty draft, and the drafter is reset for the next round.
    {
        Drafter t(std::make_unique<ThrowingDraft>(), 4);
        std::string n;
        check("a failing drafter answers nothing", t.draft({1, 2}, 3, 4, n).empty());
        check("and reports the failure", n.find("no NPU") != std::string::npos, n);
        check("and resets itself", t.resets() == 1);
    }

    // A script that runs out mid-round returns what it had (a legal, shorter draft).
    {
        const auto path = write_script("7\n8\n");
        std::string err, n;
        Drafter s(onebit::bridge::script_draft(path.string(), 0, 0, err), 4);
        std::vector<int> got = s.draft({1, 2}, 3, 4, n);
        check("an exhausted script gives a short draft", got == std::vector<int>({7, 8}), show(got));
        std::filesystem::remove(path);
    }
}

void test_server() {
    const auto path = write_script("5\n6\n7\n");
    std::string err;
    auto model = onebit::bridge::script_draft(path.string(), 0, 0, err);
    check("the server's drafter is built", model != nullptr, err);
    if (!model) return;
    Server srv(std::make_unique<Drafter>(std::move(model), 4), 0);
    if (!srv.start(err)) {
        check("the server starts", false, err);
        return;
    }
    check("the server starts", true);

    httplib::Client cli("127.0.0.1", srv.port());
    cli.set_read_timeout(5, 0);
    const auto post = [&](const std::string& body) {
        auto r = cli.Post("/v1/draft", body, "application/json");
        return r ? r->body : std::string("<no response>");
    };

    check("the endpoint is on loopback", srv.addr() == "127.0.0.1:" + std::to_string(srv.port()), srv.addr());
    check("begin is answered", post(R"({"op":"begin","seq":0,"prompt":[1,2]})").find("\"ok\":true") != std::string::npos);
    check("accept is answered", post(R"({"op":"accept","seq":0,"n":2})").find("\"ok\":true") != std::string::npos);
    check("reset is answered", post(R"({"op":"reset","seq":0})").find("\"ok\":true") != std::string::npos);

    const std::string drafted = post(R"({"op":"draft","seq":0,"n_past":3,"id_last":4,"prompt":[1,2,3],"n_max":3})");
    check("a draft request returns the script's ids", drafted.find("[5,6,7]") != std::string::npos, drafted);
    check("the first round is over", post(R"({"op":"draft","seq":0,"n_past":7,"id_last":8,"prompt":[1,2,3,4,5,6,7],"n_max":3})")
                                           .find("\"draft\":[]") != std::string::npos);

    auto r = cli.Post("/v1/draft", R"({"op":"draft","seq":1})", "application/json");
    check("a second sequence is refused", r && r->status == 400 && r->body.find("one sequence") != std::string::npos,
          r ? r->body : "");
    r = cli.Post("/v1/draft", R"({"op":"nonsense"})", "application/json");
    check("an unknown op is refused", r && r->status == 400, r ? r->body : "");
    r = cli.Post("/v1/draft", "not json", "application/json");
    check("bad JSON is refused", r && r->status == 400, r ? r->body : "");

    // The endpoint has no authentication, so a page that resolved its own name to
    // 127.0.0.1 must not be able to drive it (app/http_guard.h).
    r = cli.Post("/v1/draft", httplib::Headers{{"Host", "rebind.example"}}, R"({"op":"reset"})", "application/json");
    check("a foreign Host is refused", r && r->status == 403, r ? r->body : "");

    srv.stop();
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_script();
    test_draft_and_resync();
    test_clamps_and_errors();
    test_server();
    std::printf("%s: %d failure(s)\n", fails ? "FAILED" : "PASSED", fails);
    return fails;
}
