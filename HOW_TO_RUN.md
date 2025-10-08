# How to Run Tests from Scratch

## 🎯 TL;DR - Quick Answer

**To produce all results from scratch, simply run:**

```bash
python run_tests.py
```

That's it! This single command will:
- ✅ Test Python 3.14 official version
- ✅ Test Python 3.14 free-threaded version  
- ✅ Generate performance visualizations
- ✅ Run comparison analysis
- ✅ Copy plots to main directory

## 📋 Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Verify Python 3.14 versions**:
   ```bash
   uv python list | grep "3.14"
   ```
   Should show both official and free-threaded versions.

## 🚀 Complete Workflow

### Step 1: Clone and Setup
```bash
git clone <repository-url>
cd freethreaded-python
```

### Step 2: Run All Tests
```bash
python run_tests.py
```

### Step 3: View Results
- **Plots**: `article_images/plots/` - Performance visualizations
- **Detailed Results**: `docs/RESULTS.md` - Comprehensive analysis
- **Test Logs**: `article_images/logs/` - Debug information

## 🔧 Alternative Commands

### Run Specific Tests Only
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

### Manual Step-by-Step (if needed)
```bash
# 1. Test Python 3.14 official
uv run --python cpython-3.14.0-macos-aarch64-none python tests/test_orchestrator.py --config tests/test_config_python314.yml --test all

# 2. Test Python 3.14 free-threaded
/Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t tests/test_freethreaded_basic.py

# 3. Run comparison
python scripts/simple_comparison.py

# 4. Copy plots (if needed)
cp article_images/results_official/plots/*.png article_images/plots/
```

## 📊 Expected Output Structure

After running, you'll have:

```
article_images/
├── plots/                          # 🎨 Performance visualizations
│   ├── performance_comparison.png  # GIL vs Free-threaded comparison
│   ├── resource_utilization.png   # CPU and threading metrics  
│   └── metrics_table.png          # Detailed metrics table
├── logs/                          # 📝 Test logs
└── results_official/              # 📊 Detailed test results
    ├── results/
    │   ├── test_results.json      # Raw test data
    │   ├── test_summary.csv       # Summary CSV
    │   └── test_report.md         # Detailed report
    └── plots/                     # Generated plots
```

## 🎯 Key Results You'll See

- **Threading Speedup**: 0.28x (official) vs 0.94x (free-threaded) = **3.4x improvement**
- **Training Time**: ~0.18s (official) vs ~0.07s (theoretical free-threaded) = **2.5x faster**
- **CPU Usage**: Real measurements from actual threading tests
- **Package Compatibility**: Full support (official) vs limited (free-threaded)

## 🚨 Troubleshooting

### "uv not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

### "Python version not found"
```bash
uv python install 3.14
uv python install 3.14.0a5+freethreaded
```

### "Permission denied" on free-threaded Python
The path might be different. Check with:
```bash
uv python list
```

### "Package build failed" (free-threaded)
This is expected! The free-threaded version has limited package compatibility. The basic tests will still work.

## ✅ Success Indicators

You'll know it worked when you see:
- ✅ All tests show "PASSED" status
- 📊 Three PNG files in `article_images/plots/`
- 📝 Detailed results in `docs/RESULTS.md`
- 🎉 "All tests completed successfully!" message

## 🎉 That's It!

The `python run_tests.py` command is all you need to reproduce the complete results from scratch. Everything else is handled automatically!
