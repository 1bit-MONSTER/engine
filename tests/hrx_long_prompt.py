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
"""Check Qwen3-0.6B HRX retrieval across a multi-ubatch prompt.

Run against a llama-server built with HRX on Strix Halo:
    python3 tests/hrx_long_prompt.py [--url http://127.0.0.1:8080]
"""
import argparse
import json
import urllib.request

MIN_PROMPT_TOKENS = 4096
NEEDLE_LINE = 203
EXPECTED_ANSWER = "7341"


def build_prompt():
    lines = []
    for number in range(1, 321):
        if number == NEEDLE_LINE:
            lines.append(
                f"Record {number:03d}: The secret vault code is {EXPECTED_ANSWER}; "
                "store this fact for the final question."
            )
        else:
            lines.append(
                f"Record {number:03d}: This archived inventory lists ordinary folders "
                "and routine paper records, with no vault credentials."
            )
    return (
        "Read all 320 records carefully. Answer the final question using only the value "
        "from the record containing the secret vault code.\n\n"
        + "\n".join(lines)
        + "\n\nWhat is the secret vault code? Answer with the number only."
    )


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3-0.6B-Q4_K_M.gguf")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    prompt = build_prompt()

    tokenized = post_json(base_url + "/tokenize", {"content": prompt})
    token_count = len(tokenized.get("tokens", []))
    if token_count < MIN_PROMPT_TOKENS:
        raise SystemExit(
            f"FAIL prompt has {token_count} tokens; need at least {MIN_PROMPT_TOKENS}"
        )

    post_json(
        base_url + "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say hello and offer assistance."}],
            "temperature": 0,
            "max_tokens": 16,
            "cache_prompt": False,
            "enable_thinking": False,
        },
    )
    result = post_json(
        base_url + "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
            "cache_prompt": False,
            "enable_thinking": False,
        },
    )
    answer = result["choices"][0]["message"].get("content", "").strip()
    print(f"{'PASS' if answer == EXPECTED_ANSWER else 'FAIL'} "
          f"{token_count}-token prompt: expected {EXPECTED_ANSWER}, got {answer!r}")
    return 0 if answer == EXPECTED_ANSWER else 1


if __name__ == "__main__":
    raise SystemExit(main())
