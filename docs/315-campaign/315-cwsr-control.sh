#!/usr/bin/env bash
# engine#315 H1 CONTROL: fresh boot with amdgpu.cwsr_enable=1, identical campaign.
#
# WHY THIS EXISTS. The cwsr_enable=0 result (0 faults in 11 in-regime arms vs a 26%
# multi-session baseline) is confounded: the reboot that set cwsr_enable=0 ALSO changed box
# state (fresh boot, unfragmented device memory, no prior campaigns). Without a fresh-boot
# cwsr_enable=1 control, a clean result cannot be attributed to CWSR at all. This issue has
# already been burned once by exactly that error ("the rate followed memory-pressure
# settings"), so the control is mandatory before any causal claim.
#
# WHAT IT DOES: removes the cwsr drop-in, reboots, and (after the box returns) prints the
# exact command to run the SAME campaign. It does NOT launch the campaign itself - a reboot
# ends this process.
#
# HONEST LIMITATION: "identical campaign" can mean same PAIRS and same composition, not a
# literally identical session. Time of day, ambient temperature and whatever else the box
# has been doing differ. Record the conditions alongside the result; a cwsr_enable=1 rate
# near 26% kills H1, and near 0% confirms it.
set -uo pipefail

DROPIN=/etc/default/grub.d/amd_cwsr.cfg
TAG=${TAG:-315ctl1}
PAIRS=${PAIRS:-48}

echo "=== H1 control: fresh boot with cwsr_enable=1 ==="
echo "  current cwsr_enable : $(cat /sys/module/amdgpu/parameters/cwsr_enable)"
echo "  drop-in present     : $([ -e "$DROPIN" ] && echo yes || echo no)"
echo "  campaign to mirror  : 315-evict-trace.sh $PAIRS (the cwsr=0 run used 48)"
echo

if [ -n "$(pgrep -f '315-evict-trace' || true)" ]; then
  echo "ABORT: a 315 campaign is still running. Wait for it - this would kill its evidence."
  exit 3
fi
if [ -n "$(ps -eo args | grep -- '-dev HRX0' | grep -v grep || true)" ]; then
  echo "ABORT: an HRX job is running (not ours). Do not disturb it."
  exit 3
fi

echo "removing $DROPIN so cwsr_enable defaults to 1"
sudo -n rm -f "$DROPIN" || { echo "ABORT: could not remove the drop-in"; exit 3; }
sudo -n update-grub 2>&1 | tail -3

echo
echo "verify the parameter is GONE from the generated entry:"
if sudo -n grep -q "cwsr_enable" /boot/grub/grub.cfg; then
  echo "  WARNING: cwsr_enable still appears in grub.cfg:"
  sudo -n grep -m1 "cwsr_enable" /boot/grub/grub.cfg | cut -c1-140
  echo "  not rebooting."
  exit 3
fi
echo "  clean: no cwsr_enable in grub.cfg -> the next boot uses the driver default (1)"

echo
echo "=== after the box returns, run EXACTLY this ==="
echo "  ssh strixhalo 'cat /sys/module/amdgpu/parameters/cwsr_enable'   # expect 1"
echo "  ssh strixhalo 'nohup bash ~/wt/315-evict-trace.sh $PAIRS $TAG > ~/wt/$TAG-run.log 2>&1 &'"
echo
echo "then compare: baseline cwsr=1 must show ~26% (34/131 historically)."
echo "  ~26%  -> H1 DEAD, the cwsr=0 signal is real"
echo "  ~0%   -> H1 CONFIRMED, the signal was a fresh-boot effect"
echo
echo "rebooting now (this process ends here)"
sudo -n systemctl reboot
