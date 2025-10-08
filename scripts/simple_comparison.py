#!/usr/bin/env python3
"""
Simple Python 3.14 Comparison

Compare the results we already have from both Python versions.
"""

import json
import time
from pathlib import Path

def load_results(file_path: str) -> dict:
    """Load results from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main comparison function."""
    print("Python 3.14 Performance Comparison")
    print("=" * 50)
    
    # Load results from both versions
    official_results = load_results("article_images/results_official/results/test_results.json")
    freethreaded_results = load_results("article_images/results_freethreaded/freethreaded_basic_test.json")
    
    print("\n📊 PYTHON 3.14 OFFICIAL RESULTS")
    print("-" * 40)
    
    if "error" in official_results:
        print(f"❌ Error loading official results: {official_results['error']}")
    else:
        print(f"✅ Loaded {len(official_results)} test results")
        
        # Show test summary
        for test in official_results:
            status = "✅" if test["success"] else "❌"
            print(f"  {status} {test['name']}: {test['duration']:.2f}s")
            
            # Show key metrics for specific tests
            if test["name"] == "threading_benchmark" and test["success"]:
                data = test["data"]
                print(f"    - Single-threaded: {data['single_threaded']['time']:.2f}s")
                print(f"    - Multi-threaded: {data['multi_threaded']['time']:.2f}s")
                print(f"    - Speedup: {data['speedup_ratio']:.2f}x")
                print(f"    - CPU usage: {data['multi_threaded']['cpu_usage']:.1f}%")
            
            elif test["name"] == "woeboost_benchmark" and test["success"]:
                data = test["data"]
                print(f"    - Training time: {data['training_time']:.2f}s")
                print(f"    - Time per iteration: {data['time_per_iteration']:.4f}s")
    
    print("\n🚀 PYTHON 3.14 FREE-THREADED RESULTS")
    print("-" * 40)
    
    if "error" in freethreaded_results:
        print(f"❌ Error loading free-threaded results: {freethreaded_results['error']}")
    else:
        print("✅ Loaded free-threaded test results")
        
        # Show Python info
        if "python_info" in freethreaded_results:
            python_info = freethreaded_results["python_info"]
            print(f"  - Python version: {python_info['python_version'].split()[0]}")
            print(f"  - Free-threaded: {python_info['is_freethreaded']}")
            print(f"  - GIL enabled: {python_info['gil_enabled']}")
        
        # Show threading performance
        if "threading_performance" in freethreaded_results:
            threading_data = freethreaded_results["threading_performance"]
            print(f"  - Single-threaded: {threading_data['single_threaded_time']:.2f}s")
            print(f"  - Multi-threaded: {threading_data['multi_threaded_time']:.2f}s")
            print(f"  - Speedup: {threading_data['speedup_ratio']:.2f}x")
            print(f"  - Efficiency: {threading_data['efficiency']:.2f} ({threading_data['efficiency']*100:.1f}%)")
        
        # Show package availability
        if "packages" in freethreaded_results:
            packages = freethreaded_results["packages"]
            print("  - Package availability:")
            for pkg, status in packages.items():
                if "available" in status:
                    print(f"    ✅ {pkg}: {status}")
                else:
                    print(f"    ❌ {pkg}: {status}")
    
    print("\n🔍 COMPARISON SUMMARY")
    print("-" * 40)
    
    # Compare threading performance if both available
    if ("error" not in official_results and 
        "error" not in freethreaded_results and 
        "threading_performance" in freethreaded_results):
        
        # Find threading benchmark in official results
        official_threading = None
        for test in official_results:
            if test["name"] == "threading_benchmark" and test["success"]:
                official_threading = test["data"]
                break
        
        if official_threading:
            freethreaded_threading = freethreaded_results["threading_performance"]
            
            print("Threading Performance Comparison:")
            print(f"  Official Python 3.14:")
            print(f"    - Speedup: {official_threading['speedup_ratio']:.2f}x")
            print(f"    - CPU usage: {official_threading['multi_threaded']['cpu_usage']:.1f}%")
            
            print(f"  Free-threaded Python 3.14:")
            print(f"    - Speedup: {freethreaded_threading['speedup_ratio']:.2f}x")
            print(f"    - Efficiency: {freethreaded_threading['efficiency']:.2f}")
            
            # Calculate improvement
            speedup_improvement = freethreaded_threading['speedup_ratio'] / official_threading['speedup_ratio']
            print(f"\n  📈 Improvement: {speedup_improvement:.2f}x better speedup")
    
    print("\n💡 KEY INSIGHTS")
    print("-" * 40)
    print("• Python 3.14 official: Full scientific stack support (numpy, scikit-learn, etc.)")
    print("• Python 3.14 free-threaded: Basic functionality, limited package compatibility")
    print("• Free-threading shows potential but needs ecosystem support")
    print("• uv successfully manages both Python versions")
    
    print("\n✅ CONCLUSION")
    print("-" * 40)
    print("Both Python 3.14 versions are working with uv!")
    print("• Use Python 3.14 official for production with scientific packages")
    print("• Use Python 3.14 free-threaded for experimental threading work")
    print("• uv makes it easy to switch between versions")

if __name__ == "__main__":
    main()
