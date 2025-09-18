#!/usr/bin/env python3
"""
Test WoeBoost-style binning operations with free-threaded Python.
This simulates the CPU-intensive operations that would benefit from free-threading.
"""

import random
import sys
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

try:
    import pandas as pd
    from sklearn.datasets import make_classification

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def create_test_data(n_features=20, n_samples=10000):
    """Create test data similar to WoeBoost's input with categorical features."""
    # Always use pure Python data generation for fair comparison
    random.seed(42)
    features_data = {}
    
    # Calculate how many categorical features to add (8-9)
    n_categorical = 9
    n_numerical = n_features - n_categorical
    
    # Create numerical features
    for i in range(n_numerical):
        # Create different types of distributions
        if i % 3 == 0:
            # Normal distribution
            data = [random.gauss(0, 1) for _ in range(n_samples)]
        elif i % 3 == 1:
            # Uniform distribution
            data = [random.uniform(-5, 5) for _ in range(n_samples)]
        else:
            # Exponential distribution
            data = [random.expovariate(1) for _ in range(n_samples)]
        features_data[f"numerical_feature_{i}"] = data
    
    # Create categorical features (these require different processing)
    categorical_options = [
        ["A", "B", "C", "D", "E"],  # 5 categories
        ["Low", "Medium", "High"],  # 3 categories  
        ["Type1", "Type2", "Type3", "Type4"],  # 4 categories
        ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"],  # 6 categories
        ["Small", "Large"],  # 2 categories (binary)
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],  # 10 categories
        ["North", "South", "East", "West", "Central"],  # 5 regions
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],  # 12 months
        ["Premium", "Standard", "Basic", "Trial"]  # 4 subscription tiers
    ]
    
    for i in range(n_categorical):
        options = categorical_options[i]
        # Create categorical data with different distributions
        data = [random.choice(options) for _ in range(n_samples)]
        features_data[f"categorical_feature_{i}"] = data
    
    return features_data


def check_freethreading():
    """Check if free-threading is enabled."""
    # Check multiple ways to detect free-threading
    if hasattr(sys, "_is_freethreaded"):
        return getattr(sys, "_is_freethreaded", False)
    elif hasattr(sys, "_is_gil_enabled"):
        # If GIL status function exists, free-threading build with GIL disabled
        return not getattr(sys, "_is_gil_enabled", lambda: True)()
    else:
        # Fallback: assume not free-threaded if no indicators
        return False


def simulate_histogram_binning(data, n_bins=10):
    """Simulate histogram binning operation (CPU-intensive) for numerical data."""
    # Check if data is numerical or categorical
    if isinstance(data[0], str):
        # Categorical data - use frequency binning
        return simulate_categorical_binning(data)
    
    # Numerical data - use histogram binning
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / n_bins

    bin_counts = [0] * n_bins
    bin_sums = [0.0] * n_bins

    # Simulate the binning computation
    for value in data:
        if value == max_val:
            bin_idx = n_bins - 1
        else:
            bin_idx = int((value - min_val) / bin_width)
            bin_idx = max(0, min(bin_idx, n_bins - 1))

        bin_counts[bin_idx] += 1
        bin_sums[bin_idx] += value

    # Calculate bin averages (simulating WOE calculation)
    bin_averages = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_averages.append(bin_sums[i] / bin_counts[i])
        else:
            bin_averages.append(0.0)

    return bin_counts, bin_averages


def simulate_categorical_binning(data):
    """Simulate categorical binning operation (CPU-intensive) for categorical data."""
    # Count frequency of each category
    category_counts = {}
    for value in data:
        if value in category_counts:
            category_counts[value] += 1
        else:
            category_counts[value] = 1
    
    # Sort categories by frequency (this is CPU-intensive)
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create bins and calculate statistics
    bin_counts = [count for _, count in sorted_categories]
    bin_averages = []
    
    # Simulate WOE calculation for categories (CPU-intensive)
    total_count = sum(bin_counts)
    for category, count in sorted_categories:
        # Simulate complex categorical statistics
        frequency = count / total_count
        # Add some CPU-intensive computation to simulate real WOE calculation
        woe_simulation = 0.0
        for _ in range(200):  # Increased computational complexity for categoricals
            woe_simulation += frequency * (1 - frequency) ** 0.5
            # Add more complex categorical-specific computations
            woe_simulation += (frequency + 0.01) ** 0.25 * (1 - frequency + 0.01) ** 0.75
        bin_averages.append(woe_simulation)
    
    return bin_counts, bin_averages


def simulate_quantile_binning(data, n_bins=10):
    """Simulate quantile binning operation (CPU-intensive)."""
    # Check if data is numerical or categorical
    if isinstance(data[0], str):
        # Categorical data - use frequency binning instead
        return simulate_categorical_binning(data)
    
    # Numerical data - use quantile binning
    # Sort data for quantile calculation
    sorted_data = sorted(data)
    n = len(sorted_data)

    bin_edges = []
    for i in range(n_bins + 1):
        quantile = i / n_bins
        idx = int(quantile * (n - 1))
        bin_edges.append(sorted_data[idx])

    bin_counts = [0] * n_bins
    bin_sums = [0.0] * n_bins

    # Assign data to bins
    for value in data:
        bin_idx = next(
            (
                i
                for i in range(n_bins)
                if value >= bin_edges[i] and value <= bin_edges[i + 1]
            ),
            0,
        )
        bin_counts[bin_idx] += 1
        bin_sums[bin_idx] += value

    # Calculate bin averages
    bin_averages = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_averages.append(bin_sums[i] / bin_counts[i])
        else:
            bin_averages.append(0.0)

    return bin_counts, bin_averages


def process_single_feature(args):
    """Process a single feature (simulating WoeBoost's feature processing)."""
    feature_data, feature_name, binning_method, n_bins = args

    if binning_method == "histogram":
        bin_counts, bin_averages = simulate_histogram_binning(feature_data, n_bins)
    else:  # quantile
        bin_counts, bin_averages = simulate_quantile_binning(feature_data, n_bins)

    return feature_name, bin_counts, bin_averages


def run_concurrent_binning(
    features_data, binning_method="histogram", n_bins=10, max_workers=4
):
    """Run binning operations concurrently."""
    # Prepare arguments for each feature
    args_list = [
        (feature_data, feature_name, binning_method, n_bins)
        for feature_name, feature_data in features_data.items()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_feature, args) for args in args_list]
        results = [future.result() for future in futures]

    return results


def measure_woeboost_performance(
    n_iterations=50,
    n_samples=5000,
    n_features=15,
    python_type="standard",
    output_file=None,
):
    """Measure WoeBoost-style performance and save results."""
    print(f"Measuring WoeBoost-style performance with {python_type} Python...")
    print(f"Python version: {sys.version}")
    print(
        f"Parameters: {n_features} features, {n_samples} samples, {n_iterations} iterations"
    )

    is_freethreaded = check_freethreading()
    print(f"Free-threading detected: {is_freethreaded}")

    # Create test data
    features_data = create_test_data(n_features=n_features, n_samples=n_samples)

    print("Starting WoeBoost binning iterations...")
    start_time = time.time()
    iteration_times = []

    for i in range(n_iterations):
        iter_start = time.time()

        # Run concurrent binning (this is where the speedup happens)
        run_concurrent_binning(
            features_data, binning_method="histogram", n_bins=10, max_workers=4
        )

        iter_end = time.time()
        iter_time = iter_end - iter_start
        iteration_times.append(iter_time)

        # Progress reporting
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}: {iter_time:.4f}s")

    total_time = time.time() - start_time
    avg_time = total_time / n_iterations

    print(f"WoeBoost benchmark completed in {total_time:.3f}s")
    print(f"Average time per iteration: {avg_time:.4f}s")

    # Prepare results
    results_data = {
        "python_version": sys.version,
        "woeboost_version": "freethreaded-simulation",
        "is_freethreaded": is_freethreaded,
        "n_iterations": n_iterations,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_tasks": 4,  # Number of worker threads
        "total_time": total_time,
        "average_time_per_iteration": avg_time,
        "iteration_times": iteration_times,
        "min_time": min(iteration_times),
        "max_time": max(iteration_times),
        "binning_method": "histogram",
        "max_workers": 4,
    }

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {output_file}")

    return results_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Measure WoeBoost-style performance")
    parser.add_argument(
        "--python",
        choices=["standard", "freethreaded"],
        default="standard",
        help="Python type being tested",
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations to run"
    )
    parser.add_argument(
        "--samples", type=int, default=5000, help="Number of samples per feature"
    )
    parser.add_argument(
        "--features", type=int, default=15, help="Number of features to process"
    )
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    measure_woeboost_performance(
        n_iterations=args.iterations,
        n_samples=args.samples,
        n_features=args.features,
        python_type=args.python,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
