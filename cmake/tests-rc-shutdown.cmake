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
#
# engine#3361: the stray-unmount loop's `busybox fuser -k -m "$MOUNT_POINT"` SIGKILLs every process holding the mount,
# rc.shutdown itself ($$) and its parent included. Puppy's /sbin/poweroff runs /etc/rc.d/rc.shutdown as a child and only
# afterwards calls /bin/busybox reboot, so a reboot started from a /data cwd had rc.shutdown and its caller killed and
# nothing left to reboot: the box sat half shut down until Ctrl-Alt-Del (three rc1 board-run reboots).
#
# This row runs woof-CE's loop on a real process tree with a directory standing in for /data - no mount, no root, no
# reboot - against the line as woof-CE ships it (the control: the caller and rc.shutdown are SIGKILLed and busybox
# reboot is never reached) and against the hooked file, under dash and busybox ash. It is the no-reboot half of the
# fix; the reboot half needs the board.
add_test(NAME rc_shutdown_stray COMMAND sh ${CMAKE_SOURCE_DIR}/tests/rc_shutdown_stray_test.sh)
