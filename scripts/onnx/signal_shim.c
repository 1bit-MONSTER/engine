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
/*
 * Linked into ryzenai-server by scripts/build-onnx.sh (-Wl,--wrap=signal); ryzenai-server's
 * own sources are not changed.
 *
 * Why ryzenai-server does not exit on SIGTERM with the RyzenAI provider: its main() installs a
 * SIGTERM/SIGINT handler that calls g_server->stop() (httplib's Server::stop) and then
 * std::exit() from inside the handler. A process-directed signal runs on whichever thread the
 * kernel picks. With the NPU provider loaded the process has many threads (ONNX Runtime, the
 * provider, XRT), so the handler often lands on one of them, possibly inside a locked section.
 * Two exits then race: stop() makes listen() return on the main thread, main() returns and
 * calls exit() while the handler's std::exit() runs on the other thread. Both run the static
 * destructors (g_server with the model, ONNX Runtime's environment, the provider, XRT's
 * device objects), against each other and against the lock the interrupted thread holds:
 * glibc's exit is not safe to call twice at once. That is the hang in futex_wait after
 * "[Server] Shutting down..." and the double free seen on 2026-10-11. On the CPU provider the
 * process is small and the signal usually lands on the main thread, where std::exit() runs
 * alone (main() cannot return under its own handler), so it exits.
 *
 * The shim takes the signal out of signal context and leaves one exit: the first time the
 * program installs a SIGTERM or SIGINT handler (main, before the model loads and before any
 * runtime thread starts), both signals are blocked on that thread, so every later thread
 * inherits the block, and one thread of the shim's waits for them with sigwait(). On a signal
 * it calls RyzenAIServer::stop() on g_server only: listen() returns on the main thread, httplib
 * joins its worker threads, run() and main() return, and exit() runs once, on the main thread,
 * with no thread interrupted. Before g_server exists (the model still loading) it runs the
 * program's own handler, as before.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>

typedef void (*handler_t)(int);

sighandler_t __real_signal(int signum, sighandler_t handler);

/* main.cpp's std::unique_ptr<ryzenai::RyzenAIServer> g_server (one pointer) and
   ryzenai::RyzenAIServer::stop(). Weak, so CMake's compiler checks (which link this object
   into a test program) pass; build-onnx.sh checks both resolved in ryzenai-server, and without
   them the shim falls back to the program's own handler. */
extern void *g_server __attribute__((weak));
void _ZN7ryzenai13RyzenAIServer4stopEv(void *self) __attribute__((weak));
void _ZN7ryzenai13RyzenAIServerD1Ev(void *self) __attribute__((weak)); /* ~RyzenAIServer() */
void _ZdlPv(void *p);                                                   /* operator delete */

/* Runs first among the exit-time destructors (registered last, at the signal): deletes the
   server, and with it the model, while ONNX Runtime GenAI's globals, the provider's and XRT's
   still exist. Without it exit() destroys those first (they were created after g_server, during
   the model load) and the model last, under them. */
static void release_server(void) {
    void *p = __atomic_exchange_n(&g_server, NULL, __ATOMIC_SEQ_CST);
    if (!p) return;
    _ZN7ryzenai13RyzenAIServerD1Ev(p);
    _ZdlPv(p);
}

/* survives strip: the release package check (scripts/weekly.sh) looks for it */
__attribute__((used)) static const char g_marker[] = "1bit-onnx-signal-shim v1";

static handler_t g_handlers[2]; /* [0] SIGTERM, [1] SIGINT */
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static sigset_t g_set;

static void *waiter(void *arg) {
    (void)arg;
    for (;;) {
        int sig = 0;
        if (sigwait(&g_set, &sig) != 0) continue;
        void *server = &g_server && _ZN7ryzenai13RyzenAIServer4stopEv && _ZN7ryzenai13RyzenAIServerD1Ev
                           ? __atomic_load_n(&g_server, __ATOMIC_SEQ_CST) : NULL;
        handler_t h = __atomic_load_n(&g_handlers[sig == SIGTERM ? 0 : 1], __ATOMIC_SEQ_CST);
        if (h == SIG_IGN) continue;
        if (server && h != SIG_DFL && h != NULL) {
            atexit(release_server);
            _ZN7ryzenai13RyzenAIServer4stopEv(server); /* main() returns and exits */
            continue;
        }
        if (h == SIG_DFL || h == NULL) {
            /* no handler of the program's: the default action, as without the shim */
            __real_signal(sig, SIG_DFL);
            pthread_sigmask(SIG_UNBLOCK, &g_set, NULL);
            raise(sig);
            continue;
        }
        h(sig); /* still loading: the program's handler (it exits) */
    }
    return NULL;
}

static void start(void) {
    sigemptyset(&g_set);
    sigaddset(&g_set, SIGTERM);
    sigaddset(&g_set, SIGINT);
    pthread_sigmask(SIG_BLOCK, &g_set, NULL);
    pthread_t t;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_create(&t, &a, waiter, NULL);
    pthread_attr_destroy(&a);
}

sighandler_t __wrap_signal(int signum, sighandler_t handler) {
    if (signum != SIGTERM && signum != SIGINT) return __real_signal(signum, handler);
    pthread_once(&g_once, start);
    handler_t old = __atomic_exchange_n(&g_handlers[signum == SIGTERM ? 0 : 1], handler, __ATOMIC_SEQ_CST);
    return old ? old : SIG_DFL;
}
