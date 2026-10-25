#include <linux/module.h>
#include <linux/hwmon.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

static int read_fan_rpm_from_sysfs(int fan_num)
{
    char path[128];
    char buffer[32];
    struct file *filp;
    loff_t pos = 0;
    ssize_t bytes_read;
    int rpm = 0;

    snprintf(path, sizeof(path), "/sys/class/ec_su_axb35/fan%d/rpm", fan_num + 1);

    filp = filp_open(path, O_RDONLY, 0);
    if (IS_ERR(filp))
        return PTR_ERR(filp);

    bytes_read = kernel_read(filp, buffer, sizeof(buffer) - 1, &pos);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        kstrtoint(buffer, 10, &rpm);
    }

    filp_close(filp, NULL);
    return rpm;
}

static int read_temperature_from_sysfs(void)
{
    char path[] = "/sys/class/ec_su_axb35/temp1/temp";
    char buffer[32];
    struct file *filp;
    loff_t pos = 0;
    ssize_t bytes_read;
    int temp = 0;

    filp = filp_open(path, O_RDONLY, 0);
    if (IS_ERR(filp))
        return PTR_ERR(filp);

    bytes_read = kernel_read(filp, buffer, sizeof(buffer) - 1, &pos);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        kstrtoint(buffer, 10, &temp);
    }

    filp_close(filp, NULL);
    return temp;
}

static umode_t suaxb35_is_visible(const void *data, enum hwmon_sensor_types type, u32 attr, int channel)
{
    switch (type) {
    case hwmon_temp:
        if (attr == hwmon_temp_input)
            return 0444;
        break;
    case hwmon_fan:
        if (attr == hwmon_fan_input)
            return 0444;
        break;
    default:
        break;
    }
    return 0;
}

static int suaxb35_read(struct device *dev, enum hwmon_sensor_types type, u32 attr, int channel, long *val)
{
    switch (type) {
    case hwmon_temp:
        if (attr == hwmon_temp_input) {
            *val = read_temperature_from_sysfs() * 1000; // Convert to millidegrees
            return 0;
        }
        break;
    case hwmon_fan:
        if (attr == hwmon_fan_input && channel < 3) {
            *val = read_fan_rpm_from_sysfs(channel);
            return 0;
        }
        break;
    default:
        break;
    }
    return -EOPNOTSUPP;
}

static const struct hwmon_ops suaxb35_hwmon_ops = {
    .is_visible = suaxb35_is_visible,
    .read = suaxb35_read,
};

static const u32 temp_config[] = {
    HWMON_T_INPUT,
    0
};

static const u32 fan_config[] = {
    HWMON_F_INPUT,
    HWMON_F_INPUT,
    HWMON_F_INPUT,
    0
};

static const struct hwmon_channel_info temp_info = {
    .type = hwmon_temp,
    .config = temp_config,
};

static const struct hwmon_channel_info fan_info = {
    .type = hwmon_fan,
    .config = fan_config,
};

static const struct hwmon_channel_info *suaxb35_info[] = {
    &temp_info,
    &fan_info,
    NULL
};

static const struct hwmon_chip_info suaxb35_chip_info = {
    .ops = &suaxb35_hwmon_ops,
    .info = suaxb35_info,
};

static struct class *hwmon_class;
static struct device *parent_device;
static struct device *hwmon_dev;

static int __init su_axb35_hwmon_init(void)
{
    int ret;

    // Check if source files exist
    struct file *test_file = filp_open("/sys/class/ec_su_axb35/fan1/rpm", O_RDONLY, 0);
    if (IS_ERR(test_file)) {
        pr_err("ec_su_axb35 sysfs interface not found\n");
        return PTR_ERR(test_file);
    }
    filp_close(test_file, NULL);

    hwmon_class = class_create("su_axb35_hwmon");
    if (IS_ERR(hwmon_class)) {
        ret = PTR_ERR(hwmon_class);
        goto err;
    }

    parent_device = device_create(hwmon_class, NULL, MKDEV(0, 0), NULL, "su_axb35_device");
    if (IS_ERR(parent_device)) {
        ret = PTR_ERR(parent_device);
        goto err_device;
    }

    hwmon_dev = hwmon_device_register_with_info(parent_device, "su_axb35",
                                               NULL, &suaxb35_chip_info, NULL);
    if (IS_ERR(hwmon_dev)) {
        ret = PTR_ERR(hwmon_dev);
        pr_err("Failed to register hwmon device: %d\n", ret);
        goto err_hwmon;
    }

    pr_info("su_axb35 hwmon driver loaded - reading from existing ec_su_axb35 interface\n");
    return 0;

err_hwmon:
    device_destroy(hwmon_class, MKDEV(0, 0));
err_device:
    class_destroy(hwmon_class);
err:
    return ret;
}

static void __exit su_axb35_hwmon_exit(void)
{
    if (!IS_ERR_OR_NULL(hwmon_dev)) {
        hwmon_device_unregister(hwmon_dev);
    }
    if (!IS_ERR_OR_NULL(parent_device)) {
        device_destroy(hwmon_class, MKDEV(0, 0));
    }
    if (!IS_ERR_OR_NULL(hwmon_class)) {
        class_destroy(hwmon_class);
    }
    pr_info("su_axb35 hwmon driver unloaded\n");
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("mike");
MODULE_DESCRIPTION("SU-AXB35 Hardware Monitoring Driver - Bridge to existing interface");

module_init(su_axb35_hwmon_init);
module_exit(su_axb35_hwmon_exit);
