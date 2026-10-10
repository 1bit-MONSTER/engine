#!/usr/bin/env bash
# Re-verify every number quoted in the engine#315 evidence files, from the box, in one pass.
S="ssh -o ConnectTimeout=12 -o LogLevel=ERROR strixhalo"
pass=0; fail=0
chk() { # chk <label> <cmd> <expected-regex>
  local out; out=$(eval "$2" 2>&1)
  if printf '%s' "$out" | grep -qE "$3"; then printf 'PASS  %s\n' "$1"; pass=$((pass+1));
  else printf 'FAIL  %s\n      got: %s\n' "$1" "$(printf '%s' "$out" | head -2 | tr '\n' ' ')"; fail=$((fail+1)); fi
}
chk "row console: 5 pair lines"        "$S 'grep -cE \"^  pair\" ~/wt/315row-console.log'" '^5$'
chk "row console: 2 FAULT"             "$S 'grep -cE \"arm .: FAULT\" ~/wt/315row-console.log'" '^2$'
chk "row console: 3 OK"                "$S 'grep -cE \"arm .: OK\" ~/wt/315row-console.log'" '^3$'
chk "row console: every arm 98G"       "$S 'grep -oE \"gtt_start=[0-9]+G\" ~/wt/315row-console.log | sort -u | tr -d \"\n\"'" '^gtt_start=98G$'
chk "p1c guard line"                   "$S 'grep -m1 \"HRX returned NaN logits\" ~/wt/rt-det/315row-p1c-*/server.log'" 'count=154880 of 154880.*contiguous=yes.*sentinel=0'
chk "p1c fingerprint = 16 lines"       "$S 'grep -c hrx-tensor-fp ~/wt/rt-det/315row-p1c-*/server.log'" '^16$'
chk "p1c weights nan=8 first=32"       "$S 'grep -m1 \"ffn_moe_weights-4\" ~/wt/rt-det/315row-p1c-*/server.log'" 'nan=8 of 40 ne0=1 first=32 row=32'
chk "p1c up all-NaN 61440"             "$S 'grep -m1 \"ffn_moe_up-4\" ~/wt/rt-det/315row-p1c-*/server.log'" 'nan=61440 of 61440'
chk "p1c l_out all-NaN 20480"          "$S 'grep -m1 \"l_out-4 \" ~/wt/rt-det/315row-p1c-*/server.log'" 'nan=20480 of 20480'
chk "p1c weights hash"                 "$S 'grep -m1 \"ffn_moe_weights-4\" ~/wt/rt-det/315row-p1c-*/server.log'" 'hash=56753a5bd9bd2f87'
chk "p2l weights same hash"            "$S 'grep -m1 \"ffn_moe_weights-4\" ~/wt/rt-det/315row-p2l-*/server.log'" 'hash=56753a5bd9bd2f87'
chk "buffer trace: logits 619520 0x52" "$S 'grep -m1 \"size=619520\" ~/wt/rt-det/315accept-p1c-*/server.log'" 'host_visible=1.*memory_type=0x52 size=619520'
chk "buffer trace: arena 44048640"     "$S 'grep -m1 \"size=44048640\" ~/wt/rt-det/315accept-p1c-*/server.log'" 'host_visible=1.*memory_type=0x52 size=44048640'
chk "coherent cell 0x56"               "$S 'grep -m1 \"memory_type=0x56\" ~/wt/rt-det/315accept-p10l-*/server.log'" 'memory_type=0x56 size=619520'
chk "overshoot console archived 115G"  "$S 'grep -m1 -o \"gtt_start=115G\" ~/wt/315gsync2-overshoot-console.log'" '115G'
chk "differential: 1 file +102 -2"     "$S 'cd ~/wt/llama-315copy && git diff --stat f2099e9b7..HEAD -- ggml/src/ggml-hrx/ | tail -1'" '1 file changed, 102 insertions\(\+\), 2 deletions\(-?\)'
chk "routed-FFN diff empty"            "$S 'cd ~/wt/llama-315copy && git log --oneline f2099e9b7..HEAD -- ggml/src/ggml-hrx/dispatch_registration/common/dispatch-gated-mul-mat-id.cpp | wc -l'" '^0$'
chk "route_stride 3+10 in 315copy"     "$S 'grep -c route_stride ~/wt/llama-315copy/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-gated-mul-mat-id.cpp ~/wt/llama-315copy/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-mul-mat-id-common.h | tr \"\n\" \" \"'" ':3 .*:10 *$'
chk "route_stride 3+10 in 315instr"    "$S 'grep -c route_stride ~/wt/llama-315instr/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-gated-mul-mat-id.cpp ~/wt/llama-315instr/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-mul-mat-id-common.h | tr \"\n\" \" \"'" ':3 .*:10 *$'
chk "profile: blocks 1 x3"             "$S 'grep -c \"no-barrier-dispatches 92 blocks 1\" ~/wt/hrxsys-prof-run.log'" '^3$'
chk "profile: 612 dispatches/519 barriers" "$S 'grep -m1 -o \"dispatches 612 barriers 519\" ~/wt/hrxsys-prof-run.log'" 'dispatches 612 barriers 519'
chk "profile: block-boundary gaps 0"   "$S 'grep -m1 -o \"block-boundary gaps 0 us over 0\" ~/wt/hrxsys-prof-run.log'" 'block-boundary gaps 0 us over 0'
chk "acquire wait variant"             "$S 'grep -m1 \"iree_hsa_signal_wait_scacquire\" ~/wt/pin-fork-hrx/third_party/hrx-system/runtime/src/iree/hal/drivers/amdgpu/semaphore.c'" 'signal_wait_scacquire'
chk "timepoint advanced after submit"  "$S 'grep -c \"stream->timepoint = signal_value\" ~/wt/pin-fork-hrx/third_party/hrx-system/libhrx/src/libhrx/stream.c'" '^2$'
chk "instrument: anchored match"       "$S 'grep -c hrx_fp_name_listed ~/wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp'" '^2$'
chk "instrument: sync-before-read"     "$S 'grep -c \"ggml_backend_synchronize(backend);\" ~/wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp'" '^1$'
chk "instrument: dose branch"          "$S 'grep -c diagnostic_sync_dose ~/wt/llama-315copy/ggml/src/ggml-hrx/runtime/command-program-executor.cpp'" '^4$'
chk "wrapper dir present"              "$S 'test -x ~/wt/315gsync-on-bin/llama-server && echo yes'" '^yes$'
chk "MUL_MAT_ID clamps ids"            "$S 'grep -c \"%in_range = index.cmp ult, %id0, %s\" ~/wt/llama-315copy/ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom'" '^1$'
chk "argsort rank uses ordered cmpf"   "$S 'grep -c \"scalar.cmpf ogt\" ~/wt/llama-315copy/ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom'" '^1$'
echo "-----"; echo "PASS=$pass FAIL=$fail"
