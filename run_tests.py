#!/usr/bin/env python3
"""
Main test runner for Python 3.14 performance testing

This script provides an easy way to run all tests from scratch and generate results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def test_python_314_official():
    """Test Python 3.14 official version."""
    print("\n🚀 TESTING PYTHON 3.14 OFFICIAL")
    
    cmd = [
        "uv", "run", 
        "--python", "cpython-3.14.0-macos-aarch64-none",
        "python", "tests/test_orchestrator.py",
        "--config", "tests/test_config_python314.yml",
        "--output-dir", "article_images/results_official",
        "--test", "all",
        "--verbose"
    ]
    
    return run_command(cmd, "Python 3.14 Official - Full Test Suite")

def test_python_314_freethreaded():
    """Test Python 3.14 free-threaded version."""
    print("\n🚀 TESTING PYTHON 3.14 FREE-THREADED")
    
    cmd = [
        "/Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t",
        "tests/test_freethreaded_basic.py"
    ]
    
    return run_command(cmd, "Python 3.14 Free-threaded - Basic Tests")

def copy_plots_to_main():
    """Copy generated plots to main plots directory."""
    print("\n📊 COPYING PLOTS TO MAIN DIRECTORY")
    
    source_dir = Path("article_images/results_official/plots")
    target_dir = Path("article_images/plots")
    
    if not source_dir.exists():
        print("❌ Source plots directory not found")
        return False
    
    target_dir.mkdir(exist_ok=True)
    
    for plot_file in source_dir.glob("*.png"):
        target_file = target_dir / plot_file.name
        target_file.write_bytes(plot_file.read_bytes())
        print(f"✅ Copied {plot_file.name}")
    
    return True

def run_comparison():
    """Run comparison between versions."""
    print("\n📈 RUNNING COMPARISON")
    
    cmd = ["python", "scripts/simple_comparison.py"]
    return run_command(cmd, "Performance Comparison")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Python 3.14 Performance Test Runner")
    parser.add_argument(
        "--test", 
        choices=["official", "freethreaded", "both", "comparison", "all"],
        default="all",
        help="Which tests to run"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip copying plots to main directory"
    )
    
    args = parser.parse_args()
    
    print("Python 3.14 Performance Testing")
    print("=" * 50)
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not available. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Check if Python versions are available
    try:
        subprocess.run(["uv", "python", "list"], check=True, capture_output=True)
        print("✅ Python versions available")
    except subprocess.CalledProcessError:
        print("❌ Error checking Python versions")
        sys.exit(1)
    
    results = []
    
    # Run tests based on arguments
    if args.test in ["official", "both", "all"]:
        success = test_python_314_official()
        results.append(("Python 3.14 Official", success))
        
        if success and not args.skip_plots:
            copy_plots_to_main()
    
    if args.test in ["freethreaded", "both", "all"]:
        success = test_python_314_freethreaded()
        results.append(("Python 3.14 Free-threaded", success))
    
    if args.test in ["comparison", "all"]:
        success = run_comparison()
        results.append(("Performance Comparison", success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests completed successfully!")
        print("\nGenerated files:")
        print("- article_images/plots/ - Performance visualizations")
        print("- docs/RESULTS.md - Detailed results")
        print("- article_images/logs/ - Test logs")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
