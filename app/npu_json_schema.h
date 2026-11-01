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

// `response_format: {"type": "json_schema", "json_schema": {"schema": ...}}` for
// `1bit forward-serve`: an OpenAI structured-outputs schema compiled to GBNF in the dialect
// app/npu_grammar.h parses, so the answer is constrained *during* generation by the grammar
// engine this route already has, and no second constraint runtime is added.
//
// What is covered (issue #652 item 1): objects with properties / required / additionalProperties,
// arrays with items / minItems / maxItems, string, number, integer, boolean, null, a type list
// such as ["string", "null"], enum, const, and oneOf / anyOf as alternation.
//
// What is refused, loudly: every other keyword (pattern, format, minimum, maximum, $ref, allOf, ...)
// throws with the keyword in the message, because the house rule is that a request field this route
// cannot honour is a 400 and never a silently ignored key (docs/npu.md). A constraint quietly
// dropped from a schema is worse than one refused: the client believes the answer is validated.
//
// Property order is the converter's choice, not the client's: required properties first, then the
// optional ones, each optional one nested inside the previous one's presence so that no alternative
// can generate a stray comma. JSON object order is irrelevant to whoever parses the answer, and a
// fixed order keeps the grammar linear in the number of properties instead of 2^optional.
#pragma once

#include "npu_grammar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::npu {

namespace schema_detail {

// A string as a GBNF literal: the bytes are escaped for the grammar's own lexer, and anything
// outside printable ASCII becomes \xHH, which matches one byte (see the note on byte-wise matching
// in app/npu_grammar.h).
inline std::string literal(const std::string& text) {
    std::string out = "\"";
    for (const char ch : text) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '[': out += "\\["; break;
            case ']': out += "\\]"; break;
            default:
                if (c < 0x20 || c >= 0x7f) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\x%02X", c);
                    out += buf;
                } else {
                    out += char(c);
                }
        }
    }
    out += "\"";
    return out;
}

// One JSON document as a GBNF literal: json::dump() is the canonical text, escaped for the lexer.
inline std::string json_literal(const nlohmann::json& v) { return literal(v.dump()); }

}  // namespace schema_detail

// Compiles `schema` with every generated rule name prefixed by `prefix`, appending the rules to
// `sink`, and returns the name of the rule that accepts it.  The shared rules (string, number, ws,
// ...) are the caller's to append once: json_schema_to_gbnf() does that itself, and
// app/npu_tool_grammar.h composes several argument schemas into one document, where a second copy
// of those rules would be a duplicate definition.
inline std::string json_schema_rule(const nlohmann::json& schema, const std::string& prefix, std::string& sink) {
    using nlohmann::json;
    // GBNF rule names are [A-Za-z0-9-] only (app/npu_grammar.h, is_word_char): refuse a prefix that
    // would emit a grammar nothing can parse, rather than fail later inside the Grammar ctor.
    for (const char ch : prefix)
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-'))
            throw std::runtime_error(std::string("json_schema: rule prefix \"") + prefix +
                                     "\" has a character GBNF rule names do not allow");
    using schema_detail::json_literal;

    size_t next = 0;
    auto new_rule = [&](const std::string& body) {
        sink += prefix + std::to_string(next) + " ::= " + body + "\n";
        return prefix + std::to_string(next++);
    };
    auto alternation = [&](const std::vector<std::string>& parts) {
        std::string body = "( ";
        for (size_t i = 0; i < parts.size(); i++) body += (i ? " | " : "") + parts[i];
        body += " ) ws";
        return new_rule(body);
    };

    // Keywords this converter does not implement.  Listed, not omitted, so the refusal names them.
    static const char* const kRefused[] = {
        "pattern",           "format",        "minLength",     "maxLength",   "minimum",
        "maximum",           "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "uniqueItems",
        "contains",          "prefixItems",   "$ref",          "allOf",       "not",
        "if",                "then",          "else",          "propertyNames", "patternProperties",
        "dependentRequired", "dependentSchemas", "dependencies"};

    std::function<std::string(const json&)> value_rule = [&](const json& s) -> std::string {
        if (s.is_boolean()) {
            if (s.get<bool>()) return "any";
            throw std::runtime_error("json_schema: the schema `false` accepts no answer at all");
        }
        if (!s.is_object()) throw std::runtime_error("json_schema: a schema must be an object or a boolean");
        for (const char* key : kRefused)
            if (s.contains(key))
                throw std::runtime_error(std::string("json_schema: ") + key +
                                         " is not supported by this route's schema-to-GBNF converter");

        if (s.contains("enum")) {
            const json& e = s.at("enum");
            if (!e.is_array() || e.empty()) throw std::runtime_error("json_schema: enum must be a non-empty array");
            std::vector<std::string> lits;
            for (const auto& v : e) lits.push_back(json_literal(v));
            return alternation(lits);
        }
        if (s.contains("const")) return new_rule(json_literal(s.at("const")) + " ws");
        for (const char* key : {"oneOf", "anyOf"}) {
            if (!s.contains(key)) continue;
            const json& branches = s.at(key);
            if (!branches.is_array() || branches.empty())
                throw std::runtime_error(std::string("json_schema: ") + key + " must be a non-empty array");
            std::vector<std::string> parts;
            for (const auto& b : branches) parts.push_back(value_rule(b));
            return alternation(parts);
        }

        // `type` may be a name or a list of names (["string", "null"] is the usual nullable form).
        std::vector<std::string> types;
        if (s.contains("type")) {
            const json& t = s.at("type");
            if (t.is_string()) {
                types.push_back(t.get<std::string>());
            } else if (t.is_array() && !t.empty()) {
                for (const auto& x : t) {
                    if (!x.is_string()) throw std::runtime_error("json_schema: type entries must be strings");
                    types.push_back(x.get<std::string>());
                }
            } else {
                throw std::runtime_error("json_schema: type must be a string or a non-empty array of strings");
            }
        } else if (s.contains("properties") || s.contains("additionalProperties")) {
            types.push_back("object");
        } else if (s.contains("items")) {
            types.push_back("array");
        } else {
            return "any";  // an empty schema: any JSON value
        }

        std::vector<std::string> parts;
        for (const auto& type : types) {
            if (type == "string") {
                parts.push_back("string");
            } else if (type == "number") {
                parts.push_back("number");
            } else if (type == "integer") {
                parts.push_back("integer");
            } else if (type == "boolean") {
                parts.push_back("boolean");
            } else if (type == "null") {
                parts.push_back("null");
            } else if (type == "object") {
                // ---- object: "{" ws ( body )? "}" ws
                json props = json::object();
                if (s.contains("properties")) {
                    props = s.at("properties");
                    if (!props.is_object()) throw std::runtime_error("json_schema: properties must be an object");
                }
                if (s.contains("additionalProperties")) {
                    const json& extra = s.at("additionalProperties");
                    if (extra.is_object())
                        throw std::runtime_error(
                            "json_schema: additionalProperties as a schema is not supported by this route's "
                            "schema-to-GBNF converter; send false, or declare the properties");
                    if (!extra.is_boolean())
                        throw std::runtime_error("json_schema: additionalProperties must be true or false");
                }
                std::vector<std::string> required;
                if (s.contains("required")) {
                    const json& r = s.at("required");
                    if (!r.is_array()) throw std::runtime_error("json_schema: required must be an array of names");
                    for (const auto& name : r) {
                        if (!name.is_string()) throw std::runtime_error("json_schema: required entries must be strings");
                        if (!props.contains(name.get<std::string>()))
                            throw std::runtime_error("json_schema: required names \"" + name.get<std::string>() +
                                                     "\", which properties does not declare");
                        required.push_back(name.get<std::string>());
                    }
                }
                if (props.empty()) {
                    // Nothing declared: either only {} is allowed, or any object.
                    const bool closed = s.contains("additionalProperties") && s.at("additionalProperties") == false;
                    parts.push_back(closed ? new_rule("\"{\" ws \"}\" ws") : "object");
                    continue;
                }
                auto prop_text = [&](const std::string& key) {
                    return json_literal(json(key)) + " ws \":\" ws " + value_rule(props.at(key));
                };
                std::vector<std::string> req, opt;
                for (const auto& item : props.items()) {
                    const bool is_req = std::find(required.begin(), required.end(), item.key()) != required.end();
                    (is_req ? req : opt).push_back(item.key());
                }
                std::string body;
                for (size_t i = 0; i < req.size(); i++) body += (i ? " \",\" ws " : "") + prop_text(req[i]);
                if (!opt.empty()) {
                    if (!req.empty()) {
                        // A suffix after the required run: each optional slot is skippable on its own.
                        for (const auto& key : opt) body += " (\",\" ws " + prop_text(key) + ")?";
                    } else {
                        // No required fields: nest, so that a later slot's comma cannot appear after "{".
                        std::string chain;
                        for (size_t i = opt.size(); i-- > 0;) {
                            chain = (i == 0 ? "(" + prop_text(opt[0])
                                            : "(\",\" ws " + prop_text(opt[i]))
                                    + chain + ")?";
                        }
                        body = chain;
                    }
                }
                parts.push_back(new_rule(body.empty() ? "\"{\" ws \"}\" ws" : "\"{\" ws " + body + " \"}\" ws"));
            } else if (type == "array") {
                // ---- array: "[" ws ( item ("," ws item){min-1,max-1} )? "]" ws
                const std::string item = s.contains("items") ? value_rule(s.at("items")) : "any";
                auto count = [&](const char* key) {
                    const json& v = s.at(key);
                    if (!v.is_number_integer() || v.is_boolean())
                        throw std::runtime_error(std::string("json_schema: ") + key + " must be an integer");
                    return v.get<long long>();
                };
                long long lo = s.contains("minItems") ? count("minItems") : 0;
                long long hi = s.contains("maxItems") ? count("maxItems") : -1;
                if (lo < 0) throw std::runtime_error("json_schema: minItems must be zero or more");
                if (hi >= 0 && hi < lo) throw std::runtime_error("json_schema: maxItems is below minItems");
                if (lo > 2000 || hi > 2000)
                    throw std::runtime_error("json_schema: minItems and maxItems are limited to 2000");
                std::string rep;
                if (lo >= 1) {
                    rep = std::to_string(lo - 1) + "," + (hi < 0 ? "" : std::to_string(hi - 1));
                } else {
                    rep = "0," + (hi < 0 ? "" : std::to_string(hi - 1));
                }
                std::string body;
                if (lo >= 1) {
                    body = item + " (\",\" ws " + item + "){" + rep + "}";
                } else if (hi == 0) {
                    body.clear();
                } else {
                    body = "( " + item + " (\",\" ws " + item + "){" + rep + "} )?";
                }
                parts.push_back(new_rule(body.empty() ? "\"[\" ws \"]\" ws" : "\"[\" ws " + body + " \"]\" ws"));
            } else {
                throw std::runtime_error("json_schema: unknown type \"" + type + "\"");
            }
        }
        return parts.size() == 1 ? parts[0] : alternation(parts);
    };

    return value_rule(schema);
}

// Compiles an OpenAI structured-outputs schema into a complete GBNF document whose `root` accepts
// exactly the JSON instances that schema allows. Throws std::runtime_error (-> 400) for a schema it
// cannot honour, naming the keyword or the malformed part.
inline std::string json_schema_to_gbnf(const nlohmann::json& schema) {
    std::string rules;
    const std::string root = json_schema_rule(schema, "s", rules);
    return "root ::= " + root + "\n" + rules + json_shared_rules();
}

}  // namespace onebit::npu
