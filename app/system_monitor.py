"""System monitoring utilities for Jetson devices."""

import os
import time
from typing import Dict, Any


# Jetson GPU sysfs paths
_GPU_LOAD_PATH = "/sys/devices/platform/bus@0/17000000.gpu/load"
_GPU_CUR_FREQ_PATH = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq"
_GPU_MAX_FREQ_PATH = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq"


def _read_sysfs_int(path: str, default: int = 0) -> int:
    """Read an integer from a sysfs file."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return default


def get_gpu_stats() -> Dict[str, Any]:
    """
    Get GPU statistics using Jetson sysfs interfaces.

    On Jetson, nvidia-smi returns "Not Supported" for most metrics.
    Instead, we read directly from sysfs:
    - /sys/devices/platform/bus@0/17000000.gpu/load (0-1000 scale)
    - devfreq cur_freq / max_freq (in Hz)

    Jetson uses unified memory, so GPU memory = system memory.

    Returns:
        Dictionary with GPU utilization, frequency, and temperature
    """
    # GPU load: 0-1000 scale (divide by 10 for percentage)
    load_raw = _read_sysfs_int(_GPU_LOAD_PATH, 0)
    utilization = round(load_raw / 10.0, 1)

    # GPU frequency in MHz
    cur_freq_hz = _read_sysfs_int(_GPU_CUR_FREQ_PATH, 0)
    max_freq_hz = _read_sysfs_int(_GPU_MAX_FREQ_PATH, 0)
    cur_freq_mhz = cur_freq_hz // 1_000_000
    max_freq_mhz = max_freq_hz // 1_000_000

    # GPU temperature from thermal zone
    gpu_temp = _get_thermal_zone_temp("gpu-thermal")

    return {
        "utilization": utilization,
        "cur_freq_mhz": cur_freq_mhz,
        "max_freq_mhz": max_freq_mhz,
        "temperature_c": gpu_temp,
    }


def _get_thermal_zone_temp(zone_name: str) -> int:
    """Get temperature for a specific thermal zone by name."""
    thermal_path = "/sys/class/thermal"
    try:
        for zone in os.listdir(thermal_path):
            if not zone.startswith("thermal_zone"):
                continue
            type_file = os.path.join(thermal_path, zone, "type")
            temp_file = os.path.join(thermal_path, zone, "temp")
            try:
                with open(type_file, "r") as f:
                    if f.read().strip() == zone_name:
                        with open(temp_file, "r") as tf:
                            val = tf.read().strip()
                            if val:
                                return int(val) // 1000
            except (FileNotFoundError, ValueError):
                continue
    except OSError:
        pass
    return 0


def _read_proc_stat() -> tuple:
    """Read CPU idle and total jiffies from /proc/stat."""
    with open("/proc/stat", "r") as f:
        line = f.readline()
    parts = line.split()
    # cpu  user nice system idle iowait irq softirq steal
    values = [int(x) for x in parts[1:9]]
    idle = values[3] + values[4]  # idle + iowait
    total = sum(values)
    return idle, total


def get_cpu_stats() -> Dict[str, Any]:
    """
    Get CPU utilization using two-sample delta from /proc/stat.

    Takes two readings 100ms apart to compute instantaneous utilization
    rather than cumulative-since-boot.

    Returns:
        Dictionary with CPU utilization percentage
    """
    try:
        idle1, total1 = _read_proc_stat()
        time.sleep(0.1)
        idle2, total2 = _read_proc_stat()

        idle_delta = idle2 - idle1
        total_delta = total2 - total1

        if total_delta == 0:
            return {"utilization": 0}

        utilization = int(100 * (1 - idle_delta / total_delta))
        return {"utilization": max(0, min(100, utilization))}
    except Exception:
        return {"utilization": 0}


def get_memory_stats() -> Dict[str, Any]:
    """
    Get system memory statistics from /proc/meminfo.

    On Jetson, GPU and CPU share unified memory, so system RAM
    reflects total available memory for both CPU and GPU workloads.

    Returns:
        Dictionary with RAM usage in MB
    """
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()

        mem_total = 0
        mem_available = 0

        for line in lines:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) // 1024  # Convert KB to MB
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) // 1024

        mem_used = mem_total - mem_available

        return {
            "used_mb": mem_used,
            "total_mb": mem_total,
            "available_mb": mem_available,
            "utilization": int(100 * mem_used / max(mem_total, 1)),
        }
    except Exception:
        pass

    return {
        "used_mb": 0,
        "total_mb": 0,
        "available_mb": 0,
        "utilization": 0,
    }


def get_disk_stats() -> Dict[str, Any]:
    """
    Get disk usage statistics for the root filesystem.

    Returns:
        Dictionary with disk usage in GB
    """
    try:
        stat = os.statvfs("/")
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bfree * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb

        return {
            "used_gb": round(used_gb, 1),
            "total_gb": round(total_gb, 1),
            "free_gb": round(free_gb, 1),
            "utilization": int(100 * used_gb / max(total_gb, 1)),
        }
    except Exception:
        pass

    return {
        "used_gb": 0,
        "total_gb": 0,
        "free_gb": 0,
        "utilization": 0,
    }


def get_thermal_zones() -> Dict[str, int]:
    """
    Get temperature readings from thermal zones.

    Returns:
        Dictionary mapping thermal zone names to temperatures in Celsius
    """
    temps = {}
    thermal_path = "/sys/class/thermal"

    try:
        if os.path.exists(thermal_path):
            for zone in os.listdir(thermal_path):
                if zone.startswith("thermal_zone"):
                    temp_file = os.path.join(thermal_path, zone, "temp")
                    type_file = os.path.join(thermal_path, zone, "type")

                    if os.path.exists(temp_file) and os.path.exists(type_file):
                        with open(type_file, "r") as f:
                            zone_name = f.read().strip()
                        with open(temp_file, "r") as f:
                            val = f.read().strip()
                            if val:
                                temp_celsius = int(val) // 1000
                                temps[zone_name] = temp_celsius
    except Exception:
        pass

    return temps


def get_jetson_model() -> str:
    """
    Detect Jetson device model from device tree.

    Returns:
        Jetson model name or 'Unknown'
    """
    try:
        model_path = "/sys/firmware/devicetree/base/model"
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                model = f.read().strip("\x00").strip()
                if "Jetson" in model:
                    return model
    except Exception:
        pass

    return "Unknown Jetson Device"


def get_system_stats() -> Dict[str, Any]:
    """
    Get comprehensive system statistics for Jetson device.

    Returns:
        Dictionary containing all system metrics
    """
    gpu = get_gpu_stats()
    cpu = get_cpu_stats()
    memory = get_memory_stats()
    disk = get_disk_stats()
    thermal = get_thermal_zones()

    return {
        "device_model": get_jetson_model(),
        "gpu": {
            "utilization_percent": gpu["utilization"],
            "cur_freq_mhz": gpu["cur_freq_mhz"],
            "max_freq_mhz": gpu["max_freq_mhz"],
            "temperature_celsius": gpu["temperature_c"],
        },
        "cpu": {
            "utilization_percent": cpu["utilization"],
        },
        "memory": {
            "used_mb": memory["used_mb"],
            "total_mb": memory["total_mb"],
            "available_mb": memory["available_mb"],
            "utilization_percent": memory["utilization"],
        },
        "disk": {
            "used_gb": disk["used_gb"],
            "total_gb": disk["total_gb"],
            "free_gb": disk["free_gb"],
            "utilization_percent": disk["utilization"],
        },
        "thermal_zones": thermal,
    }
