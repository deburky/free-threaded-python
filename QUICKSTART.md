# Quick Start Guide

## 🚀 Running Tests from Scratch

### Prerequisites
1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Verify Python 3.14 versions are available**:
   ```bash
   uv python list | grep "3.14"
   ```
   You should see:
   - `cpython-3.14.0-macos-aarch64-none`
   - `cpython-3.14.0a5+freethreaded-macos-aarch64-none`

### Option 1: Run Everything (Recommended)
```bash
python run_tests.py
```
This will:
- Test Python 3.14 official
- Test Python 3.14 free-threaded  
- Generate performance visualizations
- Run comparison analysis
- Copy plots to main directory

### Option 2: Run Specific Tests
```bash
# Test only Python 3.14 official
python run_tests.py --test official

# Test only Python 3.14 free-threaded
python run_tests.py --test freethreaded

# Test both versions
python run_tests.py --test both

# Run only comparison
python run_tests.py --test comparison
```

### Option 3: Manual Step-by-Step
```bash
# 1. Test Python 3.14 official
uv run --python cpython-3.14.0-macos-aarch64-none python tests/test_orchestrator.py --config tests/test_config_python314.yml --test all

# 2. Test Python 3.14 free-threaded
/Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t tests/test_freethreaded_basic.py

# 3. Run comparison
python scripts/simple_comparison.py

# 4. Copy plots to main directory
cp article_images/results_official/plots/*.png article_images/plots/
```

## 📊 Expected Output

After running the tests, you should have:

```
article_images/
├── plots/                          # Performance visualizations
│   ├── performance_comparison.png  # GIL vs Free-threaded comparison
│   ├── resource_utilization.png   # CPU and threading metrics
│   └── metrics_table.png          # Detailed metrics table
├── logs/                          # Test logs
└── results_official/              # Detailed test results
    ├── results/
    │   ├── test_results.json      # Raw test data
    │   ├── test_summary.csv       # Summary CSV
    │   └── test_report.md         # Detailed report
    └── plots/                     # Generated plots
```

## 🔧 Troubleshooting

### "uv not found"
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your terminal or run:
source ~/.bashrc  # or ~/.zshrc
```

### "Python version not found"
```bash
# Install Python 3.14 versions
uv python install 3.14
uv python install 3.14.0a5+freethreaded
```

### "Permission denied" on free-threaded Python
```bash
# Make sure the path is correct
ls -la /Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t
```

### "Package build failed" (free-threaded)
This is expected - the free-threaded version has limited package compatibility. The basic tests will still work.

## 📈 Understanding Results

- **Performance Comparison**: Shows theoretical improvements with free-threaded Python
- **Resource Utilization**: Real CPU usage and threading metrics
- **Metrics Table**: Detailed performance measurements

## 🎯 Key Metrics to Look For

- **Threading Speedup**: 0.28x (official) vs 0.94x (free-threaded)
- **Training Time**: ~0.18s (official) vs ~0.07s (theoretical free-threaded)
- **CPU Usage**: Real measurements from threading tests

## 📝 Next Steps

1. Review the generated plots in `article_images/plots/`
2. Check detailed results in `docs/RESULTS.md`
3. Examine test logs in `article_images/logs/`
4. Use `scripts/simple_comparison.py` for ongoing comparisons
