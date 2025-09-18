#!/bin/bash

# WoeBoost Performance Comparison
# This script runs benchmarks on both standard and free-threaded Python

set -e

# Configuration
ITERATIONS=${1:-50}
SAMPLES=${2:-5000}
FEATURES=${3:-15}

echo "WoeBoost Performance Comparison"
echo "=================================================="
echo "Parameters:"
echo "  Iterations: $ITERATIONS"
echo "  Samples: $SAMPLES"
echo "  Features: $FEATURES"
echo ""

cd "$(dirname "$0")"
mkdir -p outputs/data outputs/plots outputs/logs

echo "Running WoeBoost-style benchmark with standard Python..."
python3 scripts/measure_woeboost_freethreaded.py \
    --python standard \
    --iterations $ITERATIONS \
    --samples $SAMPLES \
    --features $FEATURES \
    --output outputs/data/woeboost_performance_standard.json

if [ $? -eq 0 ]; then
    echo "Standard Python measurement completed"
else
    echo "Standard Python measurement failed"
    exit 1
fi

echo ""
echo "🧪 Running WoeBoost-style benchmark with free-threaded Python..."

# Use free-threaded Python directly
if command -v /Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t &> /dev/null; then
    echo "Using Python 3.14+freethreaded for WoeBoost-style benchmark..."
    /Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t scripts/measure_woeboost_freethreaded.py \
        --python freethreaded \
        --iterations $ITERATIONS \
        --samples $SAMPLES \
        --features $FEATURES \
        --output outputs/data/woeboost_performance_freethreaded.json
else
    echo "Python 3.14+freethreaded not found!"
    echo "Please install: uv python install 3.14.0a5+freethreaded"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "Free-threaded Python measurement completed"
else
    echo "Free-threaded Python measurement failed"
    exit 1
fi

echo ""
echo "Creating visualization..."
python3 -m pip install matplotlib numpy --quiet
python3 scripts/generate_plots.py

if [ $? -eq 0 ]; then
    echo "Visualization completed"
    echo ""
    echo "Data Files:"
    echo "  - outputs/data/woeboost_performance_standard.json"
    echo "  - outputs/data/woeboost_performance_freethreaded.json"
    echo "  - outputs/plots/performance_comparison.png"
    echo "  - outputs/plots/resource_utilization.png"
    echo "  - outputs/plots/metrics_table.png"
    echo ""
    echo "Performance comparison complete!"
else
    echo "Visualization failed"
    exit 1
fi