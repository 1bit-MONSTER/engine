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
# Lets a local Hyperloom 1.0.0 install run on the logged-in `claude` CLI (no API key / setup-token)
# when ONEBIT_HYPERLOOM_CLI_LOGIN=1: credential preflights pass, and single-shot Anthropic calls use
# the CLI (agent SDK) transport, which authenticates with the CLI's own login. Unsupported upstream.
import os, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/hyperloom-ws/.venv/lib/python3.14/site-packages/hyperloom")
FLAG_PY = 'os.environ.get("ONEBIT_HYPERLOOM_CLI_LOGIN") == "1"'
edits = {
    "common/llm_config.py": [
        ('''    return bool(
        _first_set_value(
            ANTHROPIC_CREDENTIAL_ENV_ORDER,
            env if env is not None else os.environ,
        )
    )''', '''    if ''' + FLAG_PY + ''':  # 1bit: logged-in claude CLI
        return True
    return bool(
        _first_set_value(
            ANTHROPIC_CREDENTIAL_ENV_ORDER,
            env if env is not None else os.environ,
        )
    )'''),
        ('''    if (source.get(CLAUDE_OAUTH_TOKEN_ENV) or "").strip():
        return ANTHROPIC_TRANSPORT_SDK
    return ""''', '''    if (source.get(CLAUDE_OAUTH_TOKEN_ENV) or "").strip():
        return ANTHROPIC_TRANSPORT_SDK
    if ''' + FLAG_PY + ''':  # 1bit: logged-in claude CLI
        return ANTHROPIC_TRANSPORT_SDK
    return ""'''),
    ],
    "inference_optimizer/cli/credentials.py": [
        ('''def _validate_credentials() -> None:
''', '''def _validate_credentials() -> None:
    if os.environ.get("ONEBIT_HYPERLOOM_CLI_LOGIN") == "1":  # 1bit: logged-in claude CLI
        return
'''),
    ],
    "inference_optimizer/cli/preflight.py": [
        ('_RAY_VERSION = "2.44.1"\n', '_RAY_VERSION = os.environ.get("RAY_VERSION", "2.44.1")  # 1bit: 2.44.1 has no py3.14 wheels\n'),
    ],
    "agents/kernel/scripts/install.sh": [
        ('''  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && has_anthropic=1\n''',
         '''  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && has_anthropic=1\n  [ "${ONEBIT_HYPERLOOM_CLI_LOGIN:-0}" = 1 ] && has_anthropic=1  # 1bit: logged-in claude CLI\n'''),
    ],
    "inference_optimizer/assets/install.1bit.sh": [
        ('''  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && has_anthropic=1\n''',
         '''  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && has_anthropic=1\n  [ "${ONEBIT_HYPERLOOM_CLI_LOGIN:-0}" = 1 ] && has_anthropic=1  # 1bit: logged-in claude CLI\n'''),
    ],
}
for f, reps in edits.items():
    p = os.path.join(root, f)
    if not os.path.exists(p):
        print("absent", f); continue
    s = open(p).read()
    if "ONEBIT_HYPERLOOM_CLI_LOGIN" in s or "1bit: 2.44.1" in s:
        print("already", f); continue
    for a, b in reps:
        if a not in s:
            print("MISSING in", f, repr(a[:70])); continue
        s = s.replace(a, b, 1)
    if f.endswith(".py") and "\nimport os\n" not in s and "import os" not in s.split("\n\n", 3)[0:3].__str__():
        s = "import os\n" + s
    open(p, "w").write(s)
    print("patched", f)
