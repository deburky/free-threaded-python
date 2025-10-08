#!/usr/bin/env python3
"""
Basic Free-threaded Python 3.14 Test

This script tests basic functionality that works with free-threaded Python,
focusing on threading performance and core Python features.
"""

import sys
import threading
import time
import json
from pathlib import Path

def test_python_version():
    """Test Python version and free-threading status."""
    print("=" * 60)
    print("PYTHON VERSION AND FREE-THREADING TEST")
    print("=" * 60)
    
    version_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "is_freethreaded": False,
        "gil_enabled": True
    }
    
    # Check for free-threading
    if hasattr(sys, '_is_gil_enabled'):
        gil_enabled = sys._is_gil_enabled()
        version_info["gil_enabled"] = gil_enabled
        version_info["is_freethreaded"] = not gil_enabled
    
    print(f"Python version: {version_info['python_version']}")
    print(f"Python executable: {version_info['python_executable']}")
    print(f"GIL enabled: {version_info['gil_enabled']}")
    print(f"Free-threaded: {version_info['is_freethreaded']}")
    
    return version_info

def test_threading_performance():
    """Test threading performance."""
    print("\n" + "=" * 60)
    print("THREADING PERFORMANCE TEST")
    print("=" * 60)
    
    def cpu_intensive_task(iterations=1000000):
        """CPU-intensive task for testing."""
        total = 0
        for i in range(iterations):
            total += i ** 2
        return total
    
    # Single-threaded test
    print("Running single-threaded test...")
    start_time = time.time()
    result_single = cpu_intensive_task()
    single_time = time.time() - start_time
    print(f"Single-threaded result: {result_single}")
    print(f"Single-threaded time: {single_time:.4f}s")
    
    # Multi-threaded test
    print("\nRunning multi-threaded test...")
    num_threads = 4
    results = []
    
    def worker():
        result = cpu_intensive_task()
        results.append(result)
    
    start_time = time.time()
    threads = []
    
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    multi_time = time.time() - start_time
    print(f"Multi-threaded result: {sum(results)}")
    print(f"Multi-threaded time: {multi_time:.4f}s")
    print(f"Number of threads: {num_threads}")
    
    # Calculate speedup
    speedup = single_time / multi_time if multi_time > 0 else 0
    efficiency = speedup / num_threads if num_threads > 0 else 0
    
    threading_info = {
        "single_threaded_time": single_time,
        "multi_threaded_time": multi_time,
        "num_threads": num_threads,
        "speedup_ratio": speedup,
        "efficiency": efficiency,
        "single_result": result_single,
        "multi_result": sum(results)
    }
    
    print(f"Speedup ratio: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.2f} ({efficiency*100:.1f}%)")
    
    return threading_info

def test_basic_packages():
    """Test basic packages that work with free-threaded Python."""
    print("\n" + "=" * 60)
    print("BASIC PACKAGE TEST")
    print("=" * 60)
    
    packages = {}
    
    # Test basic packages
    try:
        import json
        packages["json"] = "available"
        print("✅ json: available")
    except ImportError as e:
        packages["json"] = f"error: {e}"
        print(f"❌ json: {e}")
    
    try:
        import threading
        packages["threading"] = "available"
        print("✅ threading: available")
    except ImportError as e:
        packages["threading"] = f"error: {e}"
        print(f"❌ threading: {e}")
    
    try:
        import time
        packages["time"] = "available"
        print("✅ time: available")
    except ImportError as e:
        packages["time"] = f"error: {e}"
        print(f"❌ time: {e}")
    
    # Test numpy if available
    try:
        import numpy as np
        packages["numpy"] = f"available (version {np.__version__})"
        print(f"✅ numpy: available (version {np.__version__})")
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        result = np.sum(arr)
        print(f"  NumPy test: sum([1,2,3,4,5]) = {result}")
    except ImportError as e:
        packages["numpy"] = f"error: {e}"
        print(f"❌ numpy: {e}")
    
    return packages

def main():
    """Main test function."""
    print("FREE-THREADED PYTHON 3.14 BASIC TEST")
    print("=" * 60)
    
    # Run tests
    version_info = test_python_version()
    threading_info = test_threading_performance()
    packages_info = test_basic_packages()
    
    # Compile results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_info": version_info,
        "threading_performance": threading_info,
        "packages": packages_info,
        "test_success": True
    }
    
    # Save results
    output_dir = Path("article_images/results_freethreaded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "freethreaded_basic_test.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Python version: {version_info['python_version'].split()[0]}")
    print(f"- Free-threaded: {version_info['is_freethreaded']}")
    print(f"- Threading speedup: {threading_info['speedup_ratio']:.2f}x")
    print(f"- Threading efficiency: {threading_info['efficiency']:.2f}")

if __name__ == "__main__":
    main()
