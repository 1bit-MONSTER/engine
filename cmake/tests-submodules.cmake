# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Submodule checkout integrity (engine#2129). Its own family file for the reason engine#1979 gives: a registration in
# CMakeLists.txt's trailing block collides with every other test-adding PR, and this one is a new family -- nothing else
# registers here.
#
# The `--self-test` row is hermetic (two throwaway local repos, no network, no submodule of this tree), so it runs in CI.
# The `--root` mode is NOT a ctest row on purpose: it inspects a working tree's checkouts, and CI checks out at the pin by
# construction, so a row there would test the runner rather than the tree. It is for the box, before a build:
#
#     scripts/check-submodules.sh --root .
add_test(NAME submodules_self COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check-submodules.sh --self-test)
