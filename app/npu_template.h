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

// The container's own chat template, rendered (engine #963 option 2, docs/npu-chat-template.md).
//
// `app/npu_chatml.h` renders ChatML for every container, which is a prompt only a ChatML model can
// read.  #987 turned the silent half of that into a refusal: a container whose tokenizer has no
// `<|im_start|>`/`<|im_end|>` anchors answers 400 instead of being prompted with text its vocabulary
// spells as ordinary pieces.  This file is the other half -- the container says which template it
// wants, so render that one.
//
// The renderer is the pinned llama.cpp fork's own Jinja engine (`third_party/llama.cpp/common/jinja`,
// upstream's, ~6 kLOC, the one `llama-server --jinja` uses).  The alternative -- a subset interpreter
// written here -- would be a second, weaker implementation of exactly the component that decides what
// the model is asked, and the templates we ship use `namespace`, `loop`, negative slicing
// (`messages[::-1]`), string methods and macros: the golden tests in tests/npu_template_test.cpp are
// byte-compared against Jinja2 itself, which a subset could not pass in general.
//
// One property is load-bearing: **no approximation.**  A template that cannot be parsed or executed is
// a `TemplateError` naming the template's sha256 -- never a fall back to ChatML.  Falling back to a
// different template is the failure #963 exists to remove, so the route turns this into a 400.
//
// The renderer returns a string, and the route tokenizes it exactly as it tokenized the ChatML prompt
// (added tokens matched, `npu::Tokenizer::encode`).  The engine also marks which bytes came from the
// request (`is_input`), and a stricter route could encode request bytes with added-token matching off so
// a message spelling `<|im_end|>` cannot become the control token -- measured here, that breaks a real
// template: Zyphra/ZR1-1.5B renders `{{bos_token}}{{ns.system_prompt}}`, whose concatenation the engine
// marks input as a whole, so the BOS literalized and the served prompt came out 32 tokens where the
// container's own template asks for 21.  The llama.cpp route this engine shares its renderer with
// tokenizes the whole string too, so this route matches it and the request-text/special-token question
// stays a separate one (docs/npu-chat-template.md).
//
// When the engine is built without the fork's sources (`ONEBIT_JINJA=OFF`, e.g. a build from a tree
// whose submodules were never initialised) `renderer_built()` is false and the route keeps #987's
// behaviour -- ChatML with anchors, else a 400 -- and says so at load, rather than silently rendering
// something different from a build that has the renderer.
#pragma once

#include "template_pin.h"  // container_chat_template_text, the same read the pin and the load line use

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

#if defined(ONEBIT_JINJA)
#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#endif

namespace onebit::npu {

// The template a container declares, and the identity of the bytes it declares it with.
struct ChatTemplate {
    std::string text;    // the template source, verbatim
    std::string sha256;  // sha256 of `text` (docs/serve.md "Chat template": the same number the pin names)
    std::string source;  // "tokenizer_config.json" or "chat_template.jinja"
    // A pack knob this template branches on but a request may omit.  FastFlowLM's Qwen3 packs test
    // `'user_system_prompt' is defined`, which in Jinja is a check on a *constant* string and is
    // therefore always true, so the branch is taken and the variable is concatenated: omitting it
    // raises instead of skipping the branch.  Their own server supplies it (default ""), so this
    // engine does too rather than turning a working container into a 400.
    bool wants_user_system_prompt = false;

    bool present() const { return !text.empty(); }
};

inline ChatTemplate load_chat_template(const std::string& dir) {
    ChatTemplate t;
    t.text = container_chat_template_text(dir, &t.source);
    if (!t.text.empty())
        t.sha256 = sha256_hex(reinterpret_cast<const uint8_t*>(t.text.data()), t.text.size());
    t.wants_user_system_prompt = t.text.find("user_system_prompt") != std::string::npos;
    return t;
}

// A template that could not be rendered: unparsable, or a construct/filter the engine does not
// implement.  Never a reason to render a different template -- the caller answers 400 with
// what() and names the sha256 the container declares.
struct TemplateError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Whether this build carries the renderer (see the header comment).
inline bool renderer_built() {
#if defined(ONEBIT_JINJA)
    return true;
#else
    return false;
#endif
}

// The template variables one chat request contributes, besides its messages: `chat_template_kwargs`
// as sent, the functions the request offers (`tools`), and `enable_thinking`.  `tools` is here because
// a template is the only place a model is told what it may call: a request whose tools never reach
// the template is answered in prose (MiniCPM-V 4.7 rendered 25 tokens where its template, given the
// same request, renders 292 -- engine#1788).  An empty or non-array `tools` is left out, so a
// template's `{% if tools %}` reads it as "none offered".  `body` is the request as parsed in the
// caller's key order (see render_chat_template below for why).
inline nlohmann::ordered_json request_template_vars(const nlohmann::ordered_json& body, bool enable_thinking) {
    nlohmann::ordered_json extra = nlohmann::ordered_json::object();
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
        extra = body["chat_template_kwargs"];
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) extra["tools"] = body["tools"];
    extra["enable_thinking"] = enable_thinking;
    return extra;
}

// The request's messages as a template expects them.  The OpenAI API spells an earlier call's
// `function.arguments` as a JSON *string*; a chat template is written against the object (transformers
// hands it a dict): MiniCPM-V 4.7's walks `tool_call.arguments|items`, which does not exist for a string,
// and Qwen3's `arguments | tojson` would quote the string a second time.  So a string that parses to a
// JSON object becomes that object, in the string's own key order; anything else is left as sent.
inline nlohmann::ordered_json messages_for_template(nlohmann::ordered_json messages) {
    if (!messages.is_array()) return messages;
    for (auto& m : messages) {
        if (!m.is_object() || !m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
        for (auto& call : m["tool_calls"]) {
            if (!call.is_object() || !call.contains("function") || !call["function"].is_object()) continue;
            auto& fn = call["function"];
            if (!fn.contains("arguments") || !fn["arguments"].is_string()) continue;
            nlohmann::ordered_json parsed =
                nlohmann::ordered_json::parse(fn["arguments"].get<std::string>(), nullptr, /*allow_exceptions=*/false);
            if (parsed.is_object()) fn["arguments"] = std::move(parsed);
        }
    }
    return messages;
}

// Does any message carry a PICTURE? (engine#2324)
//
// OpenAI spells an image as a content-part array: `content: [{"type":"text",...}, {"type":"image_url","image_url":{...}}]`.
// This route parses no pixels -- there is no base64 decode and no call into npu/vision_*.h -- so a request that carries one
// would otherwise reach the template as an array and be answered as if the picture were text: a wrong answer with a 200.
// The vision chain itself exists (npu/vision_tower.h, vision_prompt.h, vision_plan.h, vision_image.h, driven by
// npu/forward/minicpm_image_probe.cpp); what is missing is the serve wiring, so this refuses rather than guesses.
inline bool message_carries_an_image(const nlohmann::ordered_json& messages) {
    if (!messages.is_array()) return false;
    for (const auto& m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto& part : m["content"]) {
            if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) continue;
            const std::string type = part["type"].get<std::string>();
            // `image_url` is the OpenAI spelling; `image` is what several local servers accept. Both mean pixels.
            if (type == "image_url" || type == "image") return true;
        }
    }
    return false;
}

// Render `messages` with the container's template.
//
//   messages          the request's messages array, as the OpenAI API spells it
//   add_generation_prompt  true on the chat route: the template appends the assistant prefix
//   extra             extra template variables (chat_template_kwargs, and a pack's own knobs -- the
//                     FastFlowLM packs' `user_system_prompt` is one), applied after the defaults
//   bos_token/eos_token  the container's own strings, so `{{ bos_token }}` renders the token the
//                     tokenizer actually has (a null token in tokenizer_config.json is "")
// `messages` and `extra` are *ordered* JSON on purpose: a template serializes what it is given (a
// tools block goes through `tojson`), so the request's key order is part of the prompt.  A
// `nlohmann::json` is a std::map, i.e. alphabetically sorted, and converting one to ordered_json
// sorts it -- so the caller passes the request as parsed, the way llama-server does.
inline std::string render_chat_template(const ChatTemplate& tmpl, const nlohmann::ordered_json& messages,
                                           bool add_generation_prompt,
                                           const nlohmann::ordered_json& extra = nlohmann::ordered_json::object(),
                                           const std::string& bos_token = std::string(),
                                           const std::string& eos_token = std::string()) {
    if (!tmpl.present()) throw TemplateError("the container carries no chat template");
#if !defined(ONEBIT_JINJA)
    (void)messages;
    (void)add_generation_prompt;
    (void)extra;
    (void)bos_token;
    (void)eos_token;
    throw TemplateError("this build has no chat-template renderer (ONEBIT_JINJA=OFF, template sha256=" +
                        tmpl.sha256 + ")");
#else
    // The template's identity travels with every failure: "template 8aa40ce1… did not render" is
    // actionable, "bad request" is not.
    const std::string ident = " (template sha256=" + (tmpl.sha256.empty() ? std::string("none") : tmpl.sha256) + ")";
    try {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(tmpl.text);
        const jinja::program prog = jinja::parse_from_tokens(lexer_res);

        jinja::context ctx(tmpl.text);
        // Ordered: a template that renders `messages` iterates them in request order, and the extra
        // variables are set after the defaults so a pack's own knob wins over ours.
        nlohmann::ordered_json in = nlohmann::ordered_json::object();
        in["messages"] = messages;
        if (!bos_token.empty()) in["bos_token"] = bos_token;
        if (!eos_token.empty()) in["eos_token"] = eos_token;
        if (add_generation_prompt) in["add_generation_prompt"] = true;
        if (extra.is_object()) {
            for (const auto& [k, v] : extra.items()) in[k] = v;
        }
        if (tmpl.wants_user_system_prompt && !in.contains("user_system_prompt")) in["user_system_prompt"] = "";
        // mark_input=true: everything that came in as data (messages, tool arguments, extra request
        // fields) is marked, so the tokenizer can refuse to parse special tokens inside it.
        jinja::global_from_json(ctx, in, /*mark_input=*/true);

        jinja::runtime runtime(ctx);
        const jinja::value result = runtime.execute(prog);
        auto gathered = jinja::runtime::gather_string_parts(result);
        std::string out = gathered->as_string().str();
        if (out.empty()) throw TemplateError("the template rendered an empty prompt" + ident);
        return out;
    } catch (const TemplateError&) {
        throw;
    } catch (const std::exception& e) {
        throw TemplateError(std::string("the container's chat template did not render: ") + e.what() + ident);
    }
#endif
}

}  // namespace onebit::npu
