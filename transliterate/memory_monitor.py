#!/usr/bin/env python3
"""
Memory monitoring utility for the transliteration pipeline
"""

import psutil
import time
import os

class MemoryMonitor:
    """
    Monitor system and process memory usage
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        
    def get_memory_usage(self):
        """Get current process memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_system_memory(self):
        """Get system memory statistics"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1024 / 1024 / 1024,  # GB
            'available': mem.available / 1024 / 1024 / 1024,  # GB
            'percent': mem.percent,
            'used': mem.used / 1024 / 1024 / 1024  # GB
        }
    
    def check_memory_status(self, warn_threshold=80, critical_threshold=90):
        """
        Check memory status and return warnings
        """
        current_memory = self.get_memory_usage()
        system_mem = self.get_system_memory()
        
        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        status = {
            'process_mb': current_memory,
            'system_percent': system_mem['percent'],
            'system_available_gb': system_mem['available'],
            'peak_mb': self.peak_memory,
            'increase_mb': current_memory - self.initial_memory,
            'warning': None,
            'critical': False
        }
        
        # Check for warnings
        if system_mem['percent'] >= critical_threshold:
            status['warning'] = f"CRITICAL: System memory usage at {system_mem['percent']:.1f}%"
            status['critical'] = True
        elif system_mem['percent'] >= warn_threshold:
            status['warning'] = f"WARNING: System memory usage at {system_mem['percent']:.1f}%"
        elif system_mem['available'] < 1.0:  # Less than 1GB available
            status['warning'] = f"WARNING: Only {system_mem['available']:.1f}GB system memory available"
        
        return status
    
    def print_memory_report(self):
        """Print a detailed memory report"""
        status = self.check_memory_status()
        system_mem = self.get_system_memory()
        
        print("\n" + "="*60)
        print("📊 MEMORY USAGE REPORT")
        print("="*60)
        
        print(f"🔧 Process Memory:")
        print(f"   Current: {status['process_mb']:.1f} MB")
        print(f"   Peak: {status['peak_mb']:.1f} MB")
        print(f"   Increase: {status['increase_mb']:.1f} MB")
        
        print(f"\n💻 System Memory:")
        print(f"   Total: {system_mem['total']:.1f} GB")
        print(f"   Used: {system_mem['used']:.1f} GB ({system_mem['percent']:.1f}%)")
        print(f"   Available: {system_mem['available']:.1f} GB")
        
        if status['warning']:
            print(f"\n⚠️  {status['warning']}")
        
        if status['critical']:
            print("\n🚨 CRITICAL: Consider reducing CHUNK_SIZE or MAX_WORKERS in config.py")
        
        print("="*60)

def monitor_memory_usage(duration=60, interval=5):
    """
    Monitor memory usage for a specified duration
    """
    monitor = MemoryMonitor()
    
    print(f"🔍 Monitoring memory usage for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        status = monitor.check_memory_status()
        
        print(f"Process: {status['process_mb']:.1f}MB | "
              f"System: {status['system_percent']:.1f}% | "
              f"Available: {status['system_available_gb']:.1f}GB")
        
        if status['warning']:
            print(f"⚠️  {status['warning']}")
        
        time.sleep(interval)
    
    monitor.print_memory_report()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        monitor_memory_usage(duration)
    else:
        monitor = MemoryMonitor()
        monitor.print_memory_report() 