#!/bin/busybox sh
# Diagnostic /init: prove userspace is reachable and capture the kernel log to the stick.
# Installed instead of the real init when the image is built with BOOT_STUB=1.
/bin/busybox mount -t proc proc /proc 2>/dev/null
/bin/busybox mount -t sysfs sysfs /sys 2>/dev/null
/bin/busybox mount -t devtmpfs devtmpfs /dev 2>/dev/null
/bin/busybox mount -t devpts devpts /dev/pts 2>/dev/null
say() { echo "$*" > /dev/console 2>/dev/null; echo "$*"; }
say ""
say "=========================================================="
say "  STUB INIT RAN - userspace IS reachable on this machine"
say "  kernel: $(uname -r)   uptime: $(cat /proc/uptime | cut -d' ' -f1)s"
say "=========================================================="
# the stick is a USB disk: usb-storage only presents it as a SCSI device, and nothing here autoloads
# modules (there is no udev), so load them or there is no /dev/sda to write to
for m in xhci_pci usb_storage uas sd_mod vfat ext4; do busybox modprobe -q "$m" 2>/dev/null; done
sleep 5
busybox mdev -s 2>/dev/null
# capture the kernel log (any oops from before init is in here) onto every partition we can mount
for p in /dev/sda1 /dev/sda2 /dev/sdb1 /dev/nvme0n1p1; do
    [ -b "$p" ] || continue
    mkdir -p /mnt
    if busybox mount "$p" /mnt 2>/dev/null; then
        { echo "stub init ran: $(date -u 2>/dev/null || cat /proc/uptime) kernel $(uname -r)"; } > /mnt/stub.log 2>/dev/null
        dmesg > /mnt/dmesg.txt 2>/dev/null
        busybox sync
        say "  wrote stub.log + dmesg.txt to $p"
        busybox umount /mnt 2>/dev/null
    fi
done
say "  devices: $(ls /dev/sd* /dev/vd* 2>/dev/null | tr '\n' ' ')"
say "  rebooting in 45s so the stick can be read from the installed system"
sleep 45
busybox sync
exec busybox reboot -f
