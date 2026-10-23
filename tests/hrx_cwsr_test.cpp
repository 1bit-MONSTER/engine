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

// engine#315 candidate 1's switch (app/hrx_cwsr.h, docs/hrx.md "CWSR save-area placement"): a fake
// cwsr_enable file and a fake environment in, the variables serve adds and its log line out. No
// GPU, no amdgpu, no backend.
#include "hrx_cwsr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <unistd.h>

using namespace onebit;

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static std::string fake_sysfs(const std::string& dir, const char* name, const char* content) {
    const std::string path = dir + "/" + name;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ::write(fd, content, std::strlen(content)) < 0) std::printf("FAIL writing %s\n", path.c_str());
    if (fd >= 0) ::close(fd);
    return path;
}

static EnvLookup env_of(std::map<std::string, std::string> m) {
    return [m](const char* k) -> const char* {
        auto it = m.find(k);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

static const std::vector<std::string> kAll = {"HSA_USE_SVM=0", "HSA_USERPTR_FOR_PAGED_MEM=0", "HRX_ALLOW_NO_SVM=1"};

int main() {
    char tmpl[] = "/tmp/hrx_cwsr_test.XXXXXX";
    const std::string dir = mkdtemp(tmpl);

    // the sysfs reader: amdgpu prints the int parameter with a newline
    expect(read_cwsr_enable(fake_sysfs(dir, "on", "1\n")) == 1, "reads 1");
    expect(read_cwsr_enable(fake_sysfs(dir, "off", "0\n")) == 0, "reads 0");
    expect(read_cwsr_enable(fake_sysfs(dir, "auto", "-1\n")) == -1, "any other value is unknown");
    expect(read_cwsr_enable(fake_sysfs(dir, "junk", "yes\n")) == -1, "junk is unknown");
    expect(read_cwsr_enable(dir + "/missing") == -1, "a missing file is unknown");

    const auto none = env_of({});
    {
        // auto, CWSR on, nothing set: the three variables and the line the docs quote
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, read_cwsr_enable(dir + "/on"), true, none);
        expect(d.env == kAll, "auto + CWSR on: all three");
        expect(d.log == "HRX: CWSR is on; CWSR save area placed in GTT (engine#315)", "auto + on log: " + d.log);
    }
    {
        // auto, CWSR off (the shipped default): nothing changes, one line says so
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, read_cwsr_enable(dir + "/off"), true, none);
        expect(d.env.empty(), "auto + CWSR off: no variables");
        expect(d.log.find("CWSR is off") != std::string::npos, "auto + off log: " + d.log);
    }
    for (const char* path : {"/auto", "/junk", "/missing"}) {
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, read_cwsr_enable(dir + path), true, none);
        expect(d.env.empty(), std::string("auto + unknown (") + path + "): no variables");
        expect(d.log.find("unknown") != std::string::npos, "auto + unknown log: " + d.log);
    }
    for (const char* var : kCwsrPlacementVars) {
        // auto, CWSR on, the user set one of the three (any value): serve adds none of them
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, 1, true, env_of({{var, "1"}}));
        expect(d.env.empty(), std::string("auto + user ") + var + ": no variables");
        expect(d.log.find(var) != std::string::npos, "auto + user log names it: " + d.log);
    }
    {
        // the user set an empty value: still set by the user
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, 1, true, env_of({{"HSA_USE_SVM", ""}}));
        expect(d.env.empty(), "auto + user HSA_USE_SVM= (empty): no variables");
    }
    {
        // auto, CWSR on, an hrx-system without the gate: nothing (HSA_USE_SVM=0 would leave no HRX0)
        const auto d = decide_cwsr_placement(CwsrPlacement::Auto, 1, false, none);
        expect(d.env.empty(), "auto + no gate: no variables");
        expect(d.log.find("no HRX_ALLOW_NO_SVM gate") != std::string::npos, "auto + no gate log: " + d.log);
    }
    {
        // on: whatever cwsr_enable says
        for (int c : {1, 0, -1}) {
            const auto d = decide_cwsr_placement(CwsrPlacement::On, c, true, none);
            expect(d.env == kAll, "on: all three, cwsr " + std::to_string(c));
            expect(d.log.find("placed in GTT") != std::string::npos, "on log: " + d.log);
        }
        // on with a user variable: serve still lists all three, and Child keeps the user's value
        const auto d = decide_cwsr_placement(CwsrPlacement::On, 1, true, env_of({{"HSA_USERPTR_FOR_PAGED_MEM", "1"}}));
        expect(d.env == kAll, "on + user: all three listed");
        expect(d.log.find("kept from the environment: HSA_USERPTR_FOR_PAGED_MEM") != std::string::npos, "on + user log: " + d.log);
        bool threw = false;
        try { decide_cwsr_placement(CwsrPlacement::On, 1, false, none); } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "on without the gate refuses");
    }
    for (int c : {1, 0, -1}) {
        // off: never
        const auto d = decide_cwsr_placement(CwsrPlacement::Off, c, true, none);
        expect(d.env.empty(), "off: no variables, cwsr " + std::to_string(c));
    }
    {
        expect(parse_cwsr_placement("auto") == CwsrPlacement::Auto, "parse auto");
        expect(parse_cwsr_placement("on") == CwsrPlacement::On, "parse on");
        expect(parse_cwsr_placement("off") == CwsrPlacement::Off, "parse off");
        bool threw = false;
        try { parse_cwsr_placement("yes"); } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "parse refuses yes");
    }

    {
        // engine#315 on the SHIPPED configuration: CWSR off is what ships, so the CWSR-on branch has to
        // be reachable in a test from a CWSR-off box. ONEBIT_CWSR_ENABLE_FILE is serve's seam
        // (app/hrx_cwsr.h: cwsr_enable_path()), and this is the decision serve makes with it.
        const auto on = fake_sysfs(dir, "seam-on", "1\n");
        const auto off = fake_sysfs(dir, "seam-off", "0\n");
        ::setenv(kCwsrEnablePathEnv, on.c_str(), 1);
        expect(cwsr_enable_path() == on, "seam: the path is the file the environment names");
        expect(read_cwsr_enable(cwsr_enable_path()) == 1, "seam: CWSR on read through the seam");
        {
            const auto d = decide_cwsr_placement(CwsrPlacement::Auto, read_cwsr_enable(cwsr_enable_path()), true, none);
            expect(d.env == kAll, "seam, auto + CWSR on: all three");
            expect(d.log == "HRX: CWSR is on; CWSR save area placed in GTT (engine#315)", "seam, auto + on log: " + d.log);
        }
        ::setenv(kCwsrEnablePathEnv, off.c_str(), 1);
        expect(read_cwsr_enable(cwsr_enable_path()) == 0, "seam: CWSR off read through the seam");
        {
            const auto d = decide_cwsr_placement(CwsrPlacement::Auto, read_cwsr_enable(cwsr_enable_path()), true, none);
            expect(d.env.empty(), "seam, auto + CWSR off: no variables");
        }
        ::unsetenv(kCwsrEnablePathEnv);
        expect(cwsr_enable_path() == kCwsrEnablePath, "seam unset: falls back to amdgpu's parameter");
        unlink(on.c_str());
        unlink(off.c_str());
    }

    for (const char* n : {"on", "off", "auto", "junk"}) unlink((dir + "/" + n).c_str());
    rmdir(dir.c_str());
    if (failures) return 1;
    std::printf("hrx_cwsr: all checks passed\n");
    return 0;
}
