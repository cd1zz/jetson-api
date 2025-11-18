"""System monitoring utilities for Jetson devices."""

import os
import re
import subprocess
from typing import Dict, Any, Optional


def get_gpu_stats() -> Dict[str, Any]:
    """
    Get GPU statistics using nvidia-smi.

    Returns:
        Dictionary with GPU utilization, memory, and temperature
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            # Parse: "utilization, mem_used, mem_total, temp"
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                return {
                    "utilization": int(parts[0].strip()),
                    "memory_used_mb": int(parts[1].strip()),
                    "memory_total_mb": int(parts[2].strip()),
                    "temperature_c": int(parts[3].strip()),
                }
    except Exception:
        pass

    return {
        "utilization": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "temperature_c": 0,
    }


def get_cpu_stats() -> Dict[str, Any]:
    """
    Get CPU statistics from /proc/stat.

    Returns:
        Dictionary with CPU utilization percentage
    """
    try:
        with open('/proc/stat', 'r') as f:
            line = f.readline()
            # cpu  user nice system idle iowait irq softirq
            parts = line.split()
            if parts[0] == 'cpu':
                idle = int(parts[4])
                total = sum(int(x) for x in parts[1:8])
                # Simple approximation - would need two samples for accuracy
                utilization = int(100 * (1 - idle / max(total, 1)))
                return {"utilization": min(utilization, 100)}
    except Exception:
        pass

    return {"utilization": 0}


def get_memory_stats() -> Dict[str, Any]:
    """
    Get system memory statistics from /proc/meminfo.

    Returns:
        Dictionary with RAM usage in MB
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()

        mem_total = 0
        mem_available = 0

        for line in lines:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) // 1024  # Convert KB to MB
            elif line.startswith('MemAvailable:'):
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
        stat = os.statvfs('/')
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
    thermal_path = '/sys/class/thermal'

    try:
        if os.path.exists(thermal_path):
            for zone in os.listdir(thermal_path):
                if zone.startswith('thermal_zone'):
                    temp_file = os.path.join(thermal_path, zone, 'temp')
                    type_file = os.path.join(thermal_path, zone, 'type')

                    if os.path.exists(temp_file) and os.path.exists(type_file):
                        with open(type_file, 'r') as f:
                            zone_name = f.read().strip()
                        with open(temp_file, 'r') as f:
                            temp_millidegrees = int(f.read().strip())
                            temp_celsius = temp_millidegrees // 1000
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
        model_path = '/sys/firmware/devicetree/base/model'
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model = f.read().strip('\x00').strip()
                # Extract relevant part (e.g., "NVIDIA Jetson AGX Orin")
                if 'Jetson' in model:
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
            "memory_used_mb": gpu["memory_used_mb"],
            "memory_total_mb": gpu["memory_total_mb"],
            "memory_utilization_percent": int(
                100 * gpu["memory_used_mb"] / max(gpu["memory_total_mb"], 1)
            ),
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
