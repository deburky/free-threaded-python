#!/usr/bin/env python3
"""
Python 3.14 Performance Comparison Script

This script runs the test suite on both Python 3.14 official and free-threaded versions
to compare performance characteristics.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

def run_command(cmd: List[str], cwd: str = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def test_python_version(python_version: str, config_file: str) -> Dict[str, Any]:
    """Test a specific Python version."""
    print(f"\n{'='*60}")
    print(f"Testing Python version: {python_version}")
    print(f"{'='*60}")
    
    # Create output directory for this version
    version_name = python_version.replace("cpython-", "").replace("-macos-aarch64-none", "")
    output_dir = f"article_images/results_{version_name}"
    
    # Run the test orchestrator
    cmd = [
        "uv", "run", 
        "--python", python_version,
        "python", "test_orchestrator.py",
        "--config", config_file,
        "--output-dir", output_dir,
        "--test", "all",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    exit_code, stdout, stderr = run_command(cmd)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Parse results
    results = {
        "python_version": python_version,
        "version_name": version_name,
        "success": exit_code == 0,
        "duration": duration,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "output_dir": output_dir
    }
    
    # Try to load test results if available
    results_file = Path(output_dir) / "test_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                test_data = json.load(f)
                results["test_data"] = test_data
        except Exception as e:
            results["test_data_error"] = str(e)
    
    return results

def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results between different Python versions."""
    comparison = {
        "summary": {},
        "detailed_comparison": {},
        "recommendations": []
    }
    
    if len(results) < 2:
        comparison["recommendations"].append("Need at least 2 Python versions for comparison")
        return comparison
    
    # Extract key metrics for comparison
    for result in results:
        version = result["version_name"]
        comparison["summary"][version] = {
            "success": result["success"],
            "duration": result["duration"],
            "exit_code": result["exit_code"]
        }
        
        # Extract test-specific data if available
        if "test_data" in result:
            test_data = result["test_data"]
            comparison["detailed_comparison"][version] = {}
            
            for test in test_data:
                if test["success"] and "data" in test:
                    comparison["detailed_comparison"][version][test["name"]] = test["data"]
    
    # Generate recommendations
    successful_results = [r for r in results if r["success"]]
    if len(successful_results) == 0:
        comparison["recommendations"].append("No Python versions completed successfully")
    elif len(successful_results) == 1:
        comparison["recommendations"].append(f"Only {successful_results[0]['version_name']} completed successfully")
    else:
        # Compare performance
        fastest = min(successful_results, key=lambda x: x["duration"])
        comparison["recommendations"].append(f"Fastest execution: {fastest['version_name']} ({fastest['duration']:.2f}s)")
    
    return comparison

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Python 3.14 Performance Comparison")
    parser.add_argument(
        "--python-versions", 
        nargs="+", 
        default=[
            "cpython-3.14.0-macos-aarch64-none",
            "cpython-3.14.0a5+freethreaded-macos-aarch64-none"
        ],
        help="Python versions to test"
    )
    parser.add_argument(
        "--config", 
        default="test_config_python314.yml",
        help="Test configuration file"
    )
    parser.add_argument(
        "--output", 
        default="python314_comparison_results.json",
        help="Output file for comparison results"
    )
    
    args = parser.parse_args()
    
    print("Python 3.14 Performance Comparison")
    print("=" * 50)
    print(f"Testing versions: {', '.join(args.python_versions)}")
    print(f"Using config: {args.config}")
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file {args.config} not found")
        sys.exit(1)
    
    # Run tests for each Python version
    all_results = []
    
    for python_version in args.python_versions:
        result = test_python_version(python_version, args.config)
        all_results.append(result)
        
        # Print summary for this version
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"\n{status} - {result['version_name']} ({result['duration']:.2f}s)")
        
        if not result["success"]:
            print(f"Error: {result['stderr'][-200:]}")  # Last 200 chars of stderr
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    comparison = compare_results(all_results)
    
    # Print summary
    print("\nSummary:")
    for version, data in comparison["summary"].items():
        status = "✅" if data["success"] else "❌"
        print(f"  {status} {version}: {data['duration']:.2f}s")
    
    # Print recommendations
    if comparison["recommendations"]:
        print("\nRecommendations:")
        for rec in comparison["recommendations"]:
            print(f"  • {rec}")
    
    # Save detailed results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_versions_tested": args.python_versions,
        "config_file": args.config,
        "results": all_results,
        "comparison": comparison
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate code
    successful_tests = sum(1 for r in all_results if r["success"])
    if successful_tests == 0:
        sys.exit(1)
    elif successful_tests < len(all_results):
        sys.exit(2)  # Partial success
    else:
        sys.exit(0)  # All successful

if __name__ == "__main__":
    main()
