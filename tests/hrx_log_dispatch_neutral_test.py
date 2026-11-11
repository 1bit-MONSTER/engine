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
"""GGML_HRX_LOG_DISPATCH stays a logging switch (engine #2143).

Measurements are taken with this flag set (the dispatch profile, the #1536 close test), so it must not be able to
change what the backend does. In the pinned llama.cpp fork the flag is read in two places, both `static void`
functions that take their arguments by const reference and return nothing: they can print, they cannot change which
registration is selected or how a value is bound. This test keeps it that way: it fails when the flag is read
anywhere else, or when one of the two readers stops being a `static void` function of const arguments.

  hrx_log_dispatch_neutral_test.py [FORK_DIR]     default: third_party/llama.cpp beside this repo's tests/
"""
import os
import re
import sys

FLAG = "GGML_HRX_LOG_DISPATCH"
SOURCE_DIRS = ("ggml/src", "src", "common", "tools")
ALLOWED_FILE = "ggml/src/ggml-hrx/dispatch_registration/dispatch-registry.cpp"
ALLOWED_FUNCTIONS = ("dispatch_log_requested_output", "dispatch_log_match")
READ = re.compile(r'getenv\s*\(\s*"%s"\s*\)' % FLAG)
FUNCTION = re.compile(r"^(static\s+)?([\w:<>\s\*&]+?)\s+(\w+)\s*\(([^;{]*)\)\s*\{", re.M)


def enclosing_function(text, offset):
    """(name, return type with qualifiers, parameter text) of the function definition that contains offset."""
    best = None
    for match in FUNCTION.finditer(text):
        if match.start() > offset:
            break
        best = match
    if best is None:
        return None
    return best.group(3), ((best.group(1) or "") + best.group(2)).split(), best.group(4)


def check_file(path, text):
    """Problems for one source file: a list of strings."""
    problems = []
    for read in READ.finditer(text):
        line = text.count("\n", 0, read.start()) + 1
        if path != ALLOWED_FILE:
            problems.append("%s:%d reads %s outside the dispatch log" % (path, line, FLAG))
            continue
        found = enclosing_function(text, read.start())
        if found is None or found[0] not in ALLOWED_FUNCTIONS:
            problems.append("%s:%d reads %s in %s, which is not a dispatch log function" %
                            (path, line, FLAG, found[0] if found else "no function"))
            continue
        name, kind, parameters = found
        if kind != ["static", "void"]:
            problems.append("%s:%d %s is '%s', not 'static void': a log function must not return a decision" %
                            (path, line, name, " ".join(kind)))
        for parameter in [p.strip() for p in parameters.split(",") if p.strip()]:
            by_value = "&" not in parameter and "*" not in parameter
            if not by_value and not parameter.startswith("const "):
                problems.append("%s:%d %s takes '%s', which it could write through" % (path, line, name, parameter))
    return problems


def check_tree(root):
    problems, reads, files = [], 0, 0
    for source_dir in SOURCE_DIRS:
        for directory, _, names in os.walk(os.path.join(root, source_dir)):
            for name in names:
                if not name.endswith((".cpp", ".h", ".c", ".hpp", ".cc")):
                    continue
                path = os.path.join(directory, name)
                with open(path, errors="replace") as handle:
                    text = handle.read()
                files += 1
                if FLAG not in text:
                    continue
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                reads += len(READ.findall(text))
                problems += check_file(relative, text)
    return problems, reads, files


def self_test():
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r, want %r" % (name, got, want))

    good = '''
static void dispatch_log_match(const DispatchRegistration & registration) {
    static const bool enabled = std::getenv("GGML_HRX_LOG_DISPATCH") != nullptr;
    if (!enabled) { return; }
}
static void dispatch_log_requested_output(const DispatchRegistration & registration, const Graph & graph,
                                          size_t node_index) {
    static const bool enabled = std::getenv("GGML_HRX_LOG_DISPATCH") != nullptr;
}
'''
    check("the two readers as they are", check_file(ALLOWED_FILE, good), [])
    check("a comment that names the flag", check_file("ggml/src/x.cpp", "// GGML_HRX_LOG_DISPATCH prints registrations\n"), [])
    check("a reader in another file", len(check_file("ggml/src/ggml-hrx/ggml-hrx.cpp", good)), 2)
    returns = good.replace("static void dispatch_log_match", "static bool dispatch_log_match")
    check("a reader that returns a value", [p.split(" is ")[1][:13] for p in check_file(ALLOWED_FILE, returns)], ["'static bool'"])
    writes = good.replace("(const DispatchRegistration & registration) {", "(DispatchMatch & match) {")
    check("a reader with a writable argument", len(check_file(ALLOWED_FILE, writes)), 1)
    elsewhere = good.replace("dispatch_log_match", "select_registration")
    check("a reader in another function", len(check_file(ALLOWED_FILE, elsewhere)), 1)
    return fails


def main():
    fails = self_test()
    for fail in fails:
        print("FAIL self-test " + fail)
    if fails:
        return 1
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                             "third_party", "llama.cpp")
    if not os.path.isfile(os.path.join(root, ALLOWED_FILE)):
        print("self-test OK; SKIP the tree check: no llama.cpp fork sources at %s" % root)
        return 0
    problems, reads, files = check_tree(root)
    for problem in problems:
        print("FAIL " + problem)
    if reads != len(ALLOWED_FUNCTIONS):
        print("FAIL %d reads of %s in %d files, expected %d (one per dispatch log function)" %
              (reads, FLAG, files, len(ALLOWED_FUNCTIONS)))
        return 1
    if problems:
        return 1
    print("OK: %s is read %d times in %d source files, both in static void log functions of const arguments" %
          (FLAG, reads, files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
