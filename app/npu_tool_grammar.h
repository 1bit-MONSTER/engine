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

// Tool calling as a grammar (issue #652 item 2): `tools`, `tool_choice` and `parallel_tool_calls`
// on `1bit forward-serve`, enforced while the answer is generated rather than parsed afterwards.
//
// This route serves ChatML models, so the call markup is Qwen3's:
//   <tool_call>{"name": "get_weather", "arguments": {...}}</tool_call>
// A request that carries `tools` gets a grammar that can only produce that shape: the name is an
// alternation over the DECLARED names, and the arguments are the function's own `parameters` schema
// compiled by app/npu_json_schema.h.  A misspelt tool, an argument of the wrong type, a missing
// required field or a Markdown fence around the call is therefore impossible, not merely unlikely --
// which is the "unforgeable" property the issue asks for.
//
// What a context-free grammar cannot express, said here rather than faked:
//   tool_choice "none"  forbidding a literal inside free text needs negative lookahead, which GBNF
//                       does not have, so "none" installs no tool grammar.  The answer is ordinary
//                       text and a caller that needs the guarantee gets it from the answer parser
//                       (#652 item 3), not from this file.
//   "auto"/"none" also skip argument-schema validation: with no grammar installed there is nothing
//                       to enforce, and refusing an otherwise serviceable request because one
//                       function's schema uses, say, `pattern` would be worse than not constraining
//                       it.  "required" does compile every schema, so there an unsupported keyword is
//                       a 400 that names it.
//   tool_choice "auto"  means "a call, or a plain answer", and a grammar cannot express that
//                       alternation without also licensing arbitrary text, so "auto" installs no
//                       tool grammar either.  A call the model does emit is only guaranteed
//                       well formed once the call grammar is installed ("required" / the object
//                       form) or the parser lands (#652 item 3).  Saying this is the point: an
//                       "auto" request must not silently become "required".
//   tool_choice "required" / {"type": "function", ...}  enforced: at least one call, and for the
//                       named function only that one.
//   parallel_tool_calls  enforced as exactly one call (false) against one or more (true).
#pragma once

#include "npu_grammar.h"
#include "npu_json_schema.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::npu {

namespace tool_detail {

struct Tool {
    std::string name;
    nlohmann::json parameters;  // the function's JSON Schema, or null when it declares none
};

// The declared tools, in the order the request lists them.  Every refusal names the field.
inline std::vector<Tool> read_tools(const nlohmann::json& tools) {
    if (!tools.is_array()) throw std::runtime_error("tools: must be an array");
    if (tools.empty()) throw std::runtime_error("tools: must not be empty");
    std::vector<Tool> out;
    for (const auto& t : tools) {
        if (!t.is_object()) throw std::runtime_error("tools: every entry must be an object");
        const std::string type = t.value("type", "function");
        if (type != "function") throw std::runtime_error("tools: type must be \"function\", not \"" + type + "\"");
        if (!t.contains("function") || !t.at("function").is_object())
            throw std::runtime_error("tools: every entry needs a function object");
        const auto& fn = t.at("function");
        if (!fn.contains("name") || !fn.at("name").is_string())
            throw std::runtime_error("tools: function needs a name");
        const std::string name = fn.at("name").get<std::string>();
        if (name.empty()) throw std::runtime_error("tools: function name must not be empty");
        for (const auto& seen : out)
            if (seen.name == name) throw std::runtime_error("tools: duplicate function name \"" + name + "\"");
        Tool tool;
        tool.name = name;
        if (fn.contains("parameters")) {
            if (!fn.at("parameters").is_object())
                throw std::runtime_error("tools: function \"" + name + "\" parameters must be an object");
            tool.parameters = fn.at("parameters");
        }
        out.push_back(std::move(tool));
    }
    return out;
}

}  // namespace tool_detail

// The grammar for one request's tools.  `tool_choice` is OpenAI's: "auto" (the default), "required",
// "none", or {"type": "function", "function": {"name": ...}}.  Returns "" when no grammar should be
// installed (no tools, or tool_choice "none"), which the caller reports as an unconstrained answer.
inline std::string tool_call_gbnf(const nlohmann::json& tools, const nlohmann::json& tool_choice,
                                  bool parallel_tool_calls) {
    using tool_detail::read_tools;
    if (!tools.is_array() || tools.empty()) return "";
    std::vector<tool_detail::Tool> list = read_tools(tools);

    // tool_choice decides which names may appear.
    std::string only;         // when set, exactly this one function
    bool required = false;    // at least one call
    if (tool_choice.is_string()) {
        const std::string mode = tool_choice.get<std::string>();
        if (mode == "none") return "";
        if (mode == "required") required = true;
        else if (mode != "auto") throw std::runtime_error("tool_choice: unknown value \"" + mode + "\"");
    } else if (tool_choice.is_object()) {
        if (tool_choice.value("type", "") != "function")
            throw std::runtime_error("tool_choice: object form needs type \"function\"");
        if (!tool_choice.contains("function") || !tool_choice.at("function").is_object() ||
            !tool_choice.at("function").contains("name"))
            throw std::runtime_error("tool_choice: object form needs function.name");
        only = tool_choice.at("function").at("name").get<std::string>();
        bool found = false;
        for (const auto& t : list) found = found || t.name == only;
        if (!found) throw std::runtime_error("tool_choice: \"" + only + "\" is not one of the tools");
        required = true;
    } else if (!tool_choice.is_null()) {
        throw std::runtime_error("tool_choice: must be a string or an object");
    }

    // Only this shape can be enforced; "none" and "auto" returned above.
    if (!required) return "";
    std::string sink;                 // the argument schemas' rules, and this file's own
    std::vector<std::string> calls;   // one alternative per allowed tool
    for (size_t i = 0; i < list.size(); i++) {
        const auto& tool = list[i];
        if (!only.empty() && tool.name != only) continue;
        // The arguments are the function's own schema, so their shape is the model's only option.
        const std::string args =
            tool.parameters.is_null() ? "any"
                                      : json_schema_rule(tool.parameters, "arg" + std::to_string(i) + "-", sink);
        const std::string call = "tc" + std::to_string(i);
        sink += call + " ::= \"{\" ws " + schema_detail::json_literal(nlohmann::json("name")) + " ws \":\" ws " +
                schema_detail::json_literal(nlohmann::json(tool.name)) + " ws \",\" ws " +
                schema_detail::json_literal(nlohmann::json("arguments")) + " ws \":\" ws " + args +
                " ws \"}\" ws\n";
        calls.push_back(call);
    }
    if (calls.empty()) throw std::runtime_error("tools: no tool is allowed by this tool_choice");

    std::string one = "( ";
    for (size_t i = 0; i < calls.size(); i++) one += (i ? " | " : "") + calls[i];
    one += " )";
    // One call inside Qwen's markup -- one or more of them when parallel calls are allowed, since
    // each alternative ends with ws so the next <tool_call> can follow directly.
    // The markup is raw text, not a JSON value: schema_detail::literal() escapes the bytes, while
    // json_literal() would also keep the quotes json::dump() adds around a string.
    std::string root = schema_detail::literal("<tool_call>") + " ws " + one + " ws " +
                       schema_detail::literal("</tool_call>") + " ws";
    if (parallel_tool_calls) root = "( " + root + " )+";

    return "root ::= " + root + "\n" + sink + json_shared_rules();
}

}  // namespace onebit::npu
