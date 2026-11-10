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
//
// tests/amdxdna_restart_harness.c - runs the amdxdna driver's own context-restart code in user space
// (engine#1750, engine#1855). tests/amdxdna_restart_test.sh applies third_party/patches/amdxdna to the pinned
// driver, cuts aie2_hwctx_job_pause(), aie2_hwctx_stop(), aie2_hwctx_restart() and aie2_hwctx_job_unpause() out
// of the resulting aie2_ctx.c, and compiles them into this file as RESTART_FUNCS. Everything they call is a stub
// here. The stubs keep the two properties the checks depend on, both read from the driver:
//   - aie2_destroy_context() clears the mailbox channel, aie2_create_context() sets it only on success;
//   - aie2_job_run() refuses a job with -ENODEV when the context has no mailbox channel, and otherwise sends it.
// What is real is the control flow under test: when the queue is unpaused, and what state the context is in then.
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint64_t u64;
struct list_head { struct list_head *next, *prev; };
static void INIT_LIST_HEAD(struct list_head *h) { h->next = h->prev = h; }
static void list_add_tail(struct list_head *n, struct list_head *h) { n->prev = h->prev; n->next = h; h->prev->next = n; h->prev = n; }
static void list_del(struct list_head *n) { n->prev->next = n->next; n->next->prev = n->prev; n->next = n->prev = n; }
static size_t list_count_nodes(struct list_head *h) { size_t c = 0; for (struct list_head *p = h->next; p != h; p = p->next) c++; return c; }
#define container_of(p, t, m) ((t *)((char *)(p) - offsetof(t, m)))
#define list_for_each_entry_safe(pos, n, head, member)                                              \
    for (pos = container_of((head)->next, __typeof__(*pos), member),                                \
         n = container_of(pos->member.next, __typeof__(*pos), member);                              \
         &pos->member != (head); pos = n, n = container_of(n->member.next, __typeof__(*n), member))

#define AMDXDNA_INVALID_ADDR (~0ULL)
struct amdxdna_gem_obj { u64 addr; struct { size_t size; } mem; };
struct amdxdna_dev { void *dev_handle; };
struct amdxdna_client { struct amdxdna_dev *xdna; int dev_heap_xa; };
struct amdxdna_hwctx_priv {
    struct amdxdna_gem_obj *heap;
    void *mbox_chann;
    void *cached_health;
    int io_lock;
    struct list_head paused_job_list;
    bool job_paused;
};
struct amdxdna_hwctx {
    struct amdxdna_client *client;
    struct amdxdna_hwctx_priv *priv;
    const char *name;
    int fw_ctx_id;
    unsigned long last_attached_heap;
};
enum job_state { PARKED, SENT, FAILED };
struct amdxdna_sched_job {
    struct amdxdna_hwctx *hwctx;
    struct list_head aie2_job_paused_list;
    enum job_state state;
    int err;
    bool sent_to_unconfigured;  // sent while the firmware context was not fully set up
};

// ---- the fake firmware -----------------------------------------------------------------------------
static struct { int create, map, config; } fail;    // what each restart step returns
static bool fw_ctx, fw_mapped, fw_configured;       // what firmware has for the context
static int destroys;
static char logbuf[4096];
static void logf_(char lvl, const char *fmt, ...)
{
    size_t n = strlen(logbuf);
    va_list ap;
    va_start(ap, fmt);
    n += (size_t)snprintf(logbuf + n, sizeof(logbuf) - n, "%c ", lvl);
    n += (size_t)vsnprintf(logbuf + n, sizeof(logbuf) - n, fmt, ap);
    snprintf(logbuf + n, sizeof(logbuf) - n, "\n");
    va_end(ap);
}
#define XDNA_DBG(xdna, fmt, ...) ((void)(xdna), logf_('D', fmt, ##__VA_ARGS__))
#define XDNA_ERR(xdna, fmt, ...) ((void)(xdna), logf_('E', fmt, ##__VA_ARGS__))
#define XDNA_WARN(xdna, fmt, ...) ((void)(xdna), logf_('W', fmt, ##__VA_ARGS__))
#define mutex_lock(m) ((void)(m))
#define mutex_unlock(m) ((void)(m))
#define kfree(p) ((void)(p))
// no extra heaps: the loop body (aie2_add_host_buf) is compiled and never entered
#define xa_for_each_range(xa, index, entry, first, last) for ((index) = (first); (void)(xa), (void)(entry), (index) < (first); (index)++)

static u64 amdxdna_obj_dma_addr(struct amdxdna_gem_obj *o) { return o->addr; }
static int aie2_create_context(void *ndev, struct amdxdna_hwctx *hwctx)
{
    (void)ndev;
    if (fail.create) return fail.create;
    hwctx->priv->mbox_chann = hwctx; hwctx->fw_ctx_id = 7; fw_ctx = true; fw_mapped = fw_configured = false;
    return 0;
}
static int aie2_destroy_context(void *ndev, struct amdxdna_hwctx *hwctx)
{
    (void)ndev;
    if (!hwctx->priv->mbox_chann) return 0;
    hwctx->priv->mbox_chann = NULL; hwctx->fw_ctx_id = -1; fw_ctx = fw_mapped = fw_configured = false; destroys++;
    return 0;
}
static int aie2_map_host_buf(void *ndev, int ctx, u64 addr, size_t size)
{
    (void)ndev; (void)ctx; (void)addr; (void)size;
    if (fail.map) return fail.map;
    fw_mapped = true;
    return 0;
}
static int aie2_add_host_buf(void *ndev, int ctx, u64 addr, size_t size) { (void)ndev; (void)ctx; (void)addr; (void)size; return 0; }
static int aie2_config_cu(struct amdxdna_hwctx *hwctx, void *cb)
{
    (void)hwctx; (void)cb;
    if (fail.config) return fail.config;
    fw_configured = true;
    return 0;
}
static void aie2_job_run(struct amdxdna_sched_job *job)
{
    if (!job->hwctx->priv->mbox_chann) { job->state = FAILED; job->err = -ENODEV; return; }
    job->state = SENT;
    job->sent_to_unconfigured = !(fw_ctx && fw_mapped && fw_configured);
}

void aie2_hwctx_job_pause(struct amdxdna_hwctx *hwctx);
void aie2_hwctx_job_unpause(struct amdxdna_hwctx *hwctx);
int aie2_hwctx_stop(struct amdxdna_hwctx *hwctx);
int aie2_hwctx_restart(struct amdxdna_dev *xdna, struct amdxdna_hwctx *hwctx);
#include RESTART_FUNCS

// ---- the scenarios ---------------------------------------------------------------------------------
static struct amdxdna_dev dev;
static struct amdxdna_client client = { .xdna = &dev };
static struct amdxdna_gem_obj heap;
static struct amdxdna_hwctx_priv priv;
static struct amdxdna_hwctx ctx;
static int fails;

static void submit(struct amdxdna_sched_job *job)   // aie2_cmd_submit()'s last step
{
    memset(job, 0, sizeof(*job));
    job->hwctx = &ctx; job->state = PARKED;
    INIT_LIST_HEAD(&job->aie2_job_paused_list);
    if (ctx.priv->job_paused) list_add_tail(&job->aie2_job_paused_list, &ctx.priv->paused_job_list);
    else aie2_job_run(job);
}
// a running context, stopped as a TDR or a suspend stops it, with one job submitted while it is stopped
static void stopped_context(struct amdxdna_sched_job *job)
{
    memset(&fail, 0, sizeof(fail)); memset(&priv, 0, sizeof(priv)); memset(&ctx, 0, sizeof(ctx));
    logbuf[0] = 0; destroys = 0;
    heap.addr = 0x1000; heap.mem.size = 4096;
    priv.heap = &heap; INIT_LIST_HEAD(&priv.paused_job_list);
    ctx.client = &client; ctx.priv = &priv; ctx.name = "hwctx.test.1";
    aie2_create_context(NULL, &ctx); fw_mapped = fw_configured = true;
    aie2_hwctx_stop(&ctx);
    destroys = 0;
    submit(job);
}
static void check(const char *scenario, bool ok, const char *what)
{
    printf("%s %s: %s\n", ok ? "ok  " : "FAIL", scenario, what);
    if (!ok) fails++;
}
static bool logged(char lvl, const char *needle)
{
    for (char *l = logbuf; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : NULL) {
        char *e = strchr(l, '\n'), *hit = strstr(l, needle);
        if (l[0] == lvl && hit && (!e || hit < e)) return true;
    }
    return false;
}
// what must hold after a restart that failed with the context still owned by a live client
static void failed_restart(const char *s, int ret, int want, struct amdxdna_sched_job *job)
{
    struct amdxdna_sched_job later;
    check(s, ret == want, "the restart returns the step's error");
    check(s, job->state == FAILED && job->err == -ENODEV, "the parked job is failed with -ENODEV by the restart, not held until close");
    check(s, !fw_ctx && !ctx.priv->mbox_chann, "firmware is left with no context for it");
    check(s, logged('E', "restart failed") && logged('E', ctx.name), "the failure is logged at error level with the context's name");
    check(s, !logged('D', "restarted") && !logged('E', "restarted"), "nothing logs it as restarted");
    submit(&later);
    check(s, later.state == FAILED && later.err == -ENODEV, "a later submission fails with -ENODEV at once");
    aie2_hwctx_job_unpause(&ctx);   // aie2_hwctx_fini() does this unconditionally
    check(s, !job->sent_to_unconfigured && !later.sent_to_unconfigured && job->state != SENT && later.state != SENT,
          "close sends nothing to firmware");
}

int main(void)
{
    struct amdxdna_sched_job job;
    int ret;

    stopped_context(&job);
    ret = aie2_hwctx_restart(&dev, &ctx);
    check("restart succeeds", ret == 0 && job.state == SENT && !job.sent_to_unconfigured && !ctx.priv->job_paused,
          "the parked job is sent to the restarted, configured context");

    stopped_context(&job); fail.config = -EIO;
    failed_restart("config CU fails", aie2_hwctx_restart(&dev, &ctx), -EIO, &job);

    stopped_context(&job); fail.map = -EINVAL;
    failed_restart("map host buffer fails", aie2_hwctx_restart(&dev, &ctx), -EINVAL, &job);

    stopped_context(&job); fail.create = -ETIMEDOUT;
    failed_restart("create context fails", aie2_hwctx_restart(&dev, &ctx), -ETIMEDOUT, &job);

    // a context whose owner is closing: -ESRCH is what the TDR walk and the device resume key on (0004, 0008)
    stopped_context(&job); heap.addr = AMDXDNA_INVALID_ADDR;
    ret = aie2_hwctx_restart(&dev, &ctx);
    check("owner is closing", ret == -ESRCH, "the restart still returns -ESRCH");
    check("owner is closing", !fw_ctx, "no firmware context is created for it");
    aie2_hwctx_job_unpause(&ctx);
    check("owner is closing", job.state == FAILED && job.err == -ENODEV, "close fails the parked job with -ENODEV");

    if (fails) { printf("%d check(s) failed\n", fails); return 1; }
    printf("ok   amdxdna restart\n");
    return 0;
}
