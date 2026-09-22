#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write chat-template golden cases rendered by HF transformers (jinja2).

usage: make_template_golden.py --model <hf_dir_or_id> --out cases.jsonl

Each line: {"messages", "tools", "add_generation_prompt", "kwargs", "expected"}.
"""
import argparse
import json

from transformers import AutoTokenizer

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}]},
    {"messages": [{"role": "user", "content": "Hello!"}], "kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": False},
    {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}]},
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nadd them\n</think>\n\n4"},
        {"role": "user", "content": "Thanks"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        {"role": "user", "content": "Thanks"},
    ]},
    {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER_TOOL]},
    {"messages": [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "{\"temp_c\": 18}"},
    ], "tools": [WEATHER_TOOL]},
    {"messages": [{"role": "user", "content": "日本語で答えて 😀 — «quotes» \"escaped\" \\ back"}]},
    {"messages": [{"role": "user", "content": "line1\nline2\n\n  indented\ttab"}]},
    {"messages": [{"role": "user", "content": ""}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.out, "w") as f:
        for case in CASES:
            case = {"tools": None, "add_generation_prompt": True, "kwargs": {}, **case}
            case["expected"] = tok.apply_chat_template(
                case["messages"], tools=case["tools"], add_generation_prompt=case["add_generation_prompt"],
                tokenize=False, **case["kwargs"])
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("wrote %d cases to %s" % (len(CASES), args.out))


if __name__ == "__main__":
    main()
