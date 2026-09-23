# Vendored: lemonade (LOCAL-ONLY source)

> **LOCAL-ONLY.** Refresh this vendored tree from the **local** lemonade source
> (`/home/bcloud/1bit-lemonade-v1170/third_party/lemonade`), never from
> `github.com/lemonade-sdk/lemonade` (see that worktree's `RULES.md`). Do NOT
> `git fetch` / `pull` / `clone`, push PRs, open issues, or run CI against
> upstream.

This snapshot is at **lemonade v11.9.0**.

## What is upstream vs local

As of v11.9.0, **upstream now carries the `llamacpp-hrx` backend itself**
(`src/cpp/server/backends/hrx/hrx_server.cpp` + `lemon/backends/hrx/`), so that
part of our HRX work is no longer a local patch — the HRX backend code is
byte-identical to upstream.

The **local-only** deltas carried on top of v11.9.0 are:

1. **`hrx-b66` pin** (newer than upstream's `hrx-b59`): in
   `src/cpp/resources/backend_versions.json` and `test/cpp/test_hrx_contract.cpp`.
2. **HRX model-registry annotations**: `src/cpp/resources/server_models.json`
   carries the `*-HRX` entries (`hrx_serve` / `hrx_token_embd` / `hrx_embd_w`),
   `tools/gen_hrx_model_entries.py` and `tools/annotate_hrx_embedding_quants.py`
   are local-only (upstream does not read or generate these).
3. **Embeddability patch** in `CMakeLists.txt` (see below).
4. **`onebit` backend** (goal mtvd3pmx R8 — the engine as a Lemonade executor):
   `src/cpp/include/lemon/backends/onebit/` (`onebit.h`, `onebit_server.h`),
   `src/cpp/server/backends/onebit/onebit_server.cpp`, the `"onebit|onebit"` line in
   `CMakeLists.txt`'s `LEMON_BACKENDS`, and the `set_registry_surface()` hook +
   private `registry_surface_` member in `include/lemon/server.h` / `server.cpp`
   (with `GET /v1/registry`). It makes native/FLM artifacts executable on the
   `--lemonade` face. **NOT upstream** — a re-vendor that overwrites `CMakeLists.txt`
   or `server.{h,cpp}` from the local source drops it, and `rsync --delete` would
   also remove the `onebit/` folders. Either land this same delta in the local
   source (`/home/bcloud/1bit-lemonade-v1170/third_party/lemonade`) before the next
   refresh, or re-apply it after.

5. **`hrx_device` option** (1bit engine, step 2): `lemon/backends/hrx/hrx.h`
   declares `hrx_device` (default `HRX0`, config key `hrx.device`), and
   `hrx_server.{h,cpp}` pass it to `--device` instead of the hard-coded `HRX0`,
   because an HRX2 build names its device `HRX20`. Upstream behavior is unchanged
   when the option is unset.

6. **`mlx` backend** (1bit engine: MLX models on Apple Silicon): `src/cpp/include/lemon/backends/mlx/`
   (`mlx.h`, `mlx_server.h`), `src/cpp/server/backends/mlx/mlx_server.cpp`, the `"mlx|mlx"` line in
   `CMakeLists.txt`'s `LEMON_BACKENDS`, and four `*-MLX` entries (`recipe: mlx`, checkpoint = an
   `mlx-community` Hugging Face id) in `src/cpp/resources/server_models.json`. The executor is the
   `server` binary of lemon-mlx-engine (fork `bong-water-water-bong/lemon-mlx-engine`, built with MLX's
   Metal backend on macOS), found through the `mlx_bin` option, then `$LEMONADE_MLX_SERVER`, then
   `lemon-mlx-server` on PATH. `load()` spawns `<server> <checkpoint> --port <p>` and waits on `/health`;
   chat/completion requests carry the checkpoint in `model`, because the MLX server selects its model
   by Hugging Face id. Supported on `macos` only. **NOT upstream**: re-apply after a re-vendor, like
   `onebit`.

> Note: the `stream_stall_timeout` config key that our v11.8.x snapshot carried
> was **dropped** in this re-vendor — v11.9.0 handles the streaming-stall bound
> via `global_timeout` (upstream #3386), and local review confirmed the extra
> config key is not needed.

```sh
# Re-vendor FROM the local source:
rsync -a --exclude=.git --exclude=UPSTREAM.md \
  /home/bcloud/1bit-lemonade-v1170/third_party/lemonade/ third_party/lemonade/
# re-apply the embeddability patch below AND the `onebit` backend delta (item 4 above)
```

## Local patch: embeddability

`CMakeLists.txt` carries one local patch (see the "Embedding" comment near
`lemonade-server-core`):

1. `CMAKE_SOURCE_DIR` → `CMAKE_CURRENT_SOURCE_DIR` everywhere — no-op when
   built standalone, fixes packaging paths when built as a subdirectory of
   the 1bit-monster repo via `add_subdirectory`.
2. Treat the parent's FetchContent-provided `nlohmann_json` and `httplib`
   targets as "system" deps (`USE_SYSTEM_JSON` / `USE_SYSTEM_HTTPLIB` set ON
   when `TARGET nlohmann_json` / `TARGET httplib` exist) so the vendored tree
   does not FetchContent a second copy and collide on target names. The
   `lemonade-httplib` interface target short-circuits to link the parent's
   `httplib` target directly when it exists.
3. PUBLIC include dirs on `lemonade-server-core` so parent targets
   (`unified_server`, `unified_router`) linking the OBJECT library see
   `lemon/` headers + generated headers (upstream uses a subdirectory-local
   `include_directories()` that does not propagate to consumers).
4. `add_test()` police guarded by `BUILD_TESTING` so it does not leak into
   the parent scope when embedded via `add_subdirectory()`.
5. `add_dependencies(lemonade-server-core copy_resources)` so the resource
   copy fires even though `lemond` (whose POST_BUILD would trigger it) is
   never built in the embed.

Drop the patch when upstream adopts any of these changes.
