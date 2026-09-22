// check.h: minimal test assertions (no framework dependency).
#pragma once

#include <cstdio>
#include <cstdlib>

inline int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

#define REQUIRE(cond)                                                      \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "%s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

inline int test_result(const char* name) {
    if (g_failures) std::fprintf(stderr, "%s: %d check(s) failed\n", name, g_failures);
    else std::printf("%s: ok\n", name);
    return g_failures ? 1 : 0;
}
