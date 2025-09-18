# Free-Threaded Python Performance Testing

A clean performance comparison suite demonstrating **6.35× speedup** from Python's GIL removal.

## Quick Start

```bash
./run_comparison.sh
```

This script:
- Runs WoeBoost-style binning operations on standard Python 3.10 (GIL enabled)  
- Runs the same operations on free-threaded Python 3.14 (GIL disabled)
- Generates performance comparison charts showing **6.35× speedup**

## Performance Results

Measurements show **6.35× speedup** with free-threaded Python for CPU-intensive multi-threaded workloads like WoeBoost histogram binning operations.

**Results:**
- Standard Python 3.10: 1.201s
- Free-threaded Python 3.14: 0.189s  
- **Speedup: 6.35× faster**

**Workload:**
- 10,000 data samples
- 15 features (9 categorical + 6 numerical)
- 4 worker threads
- 50 iterations

## Structure

```
📁 scripts/                           # Python scripts
  ├── measure_woeboost_freethreaded.py # Real WoeBoost benchmarks
  └── generate_plots.py               # Visualization generation
📁 outputs/                           # All results
  ├── data/                          # Performance JSON files  
  ├── plots/                         # Generated charts
  └── logs/                          # Log files
📄 run_comparison.sh                  # Main execution script
```

## Requirements

- Standard Python 3.10+ (for baseline comparison)
- `uv` package manager (handles dependencies automatically)
- Free-threaded Python 3.14 (install via `uv python install 3.14.0a5+freethreaded`)