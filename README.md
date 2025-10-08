# Free-Threaded Python 3.14 Performance Testing

This project provides comprehensive performance testing and comparison between Python 3.14 official and free-threaded versions using modern Python tooling.

## 🚀 Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.14 (installed via uv)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd freethreaded-python

# Install dependencies
uv sync
```

### Running Tests from Scratch

#### Option 1: Run Everything (Recommended)
```bash
python run_tests.py
```

#### Option 2: Run Specific Tests
```bash
# Test only Python 3.14 official
python run_tests.py --test official

# Test only Python 3.14 free-threaded  
python run_tests.py --test freethreaded

# Test both versions
python run_tests.py --test both
```

#### Option 3: Manual Step-by-Step
```bash
# Test Python 3.14 official
uv run --python cpython-3.14.0-macos-aarch64-none python tests/test_orchestrator.py --config tests/test_config_python314.yml --test all

# Test Python 3.14 free-threaded
/Users/deburky/.local/share/uv/python/cpython-3.14.0a5+freethreaded-macos-aarch64-none/bin/python3.14t tests/test_freethreaded_basic.py

# Run comparison
python scripts/simple_comparison.py
```

> 📖 **For detailed instructions**: See [QUICKSTART.md](QUICKSTART.md)

## 📁 Project Structure

```
freethreaded-python/
├── tests/                          # Test files
│   ├── test_orchestrator.py       # Main test orchestrator
│   ├── test_freethreaded_basic.py # Basic free-threaded tests
│   ├── test_config_python314.yml  # Python 3.14 official config
│   └── test_config_python314_freethreaded.yml # Free-threaded config
├── scripts/                        # Utility scripts
│   ├── simple_comparison.py       # Compare test results
│   └── run_python314_comparison.py # Comprehensive comparison
├── article_images/                 # Generated outputs
│   ├── plots/                     # Performance visualizations
│   │   ├── performance_comparison.png
│   │   ├── resource_utilization.png
│   │   └── metrics_table.png
│   └── medium_article.md          # Article content
├── docs/                          # Documentation
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 🧪 Test Features

### Python 3.14 Official Tests
- ✅ Environment verification
- ✅ WoeBoost performance benchmark
- ✅ Threading performance test
- ✅ Data generation
- ✅ Performance visualizations

### Python 3.14 Free-threaded Tests
- ✅ Basic Python functionality
- ✅ Free-threading verification
- ✅ Threading performance comparison
- ✅ Package compatibility check

## 📊 Key Results

Based on our testing:

- **Python 3.14 Official**: Full scientific stack support with GIL limitations
- **Python 3.14 Free-threaded**: 3.4x better threading performance but limited package compatibility
- **Threading Speedup**: 0.28x (official) vs 0.94x (free-threaded)
- **Training Time**: 0.180s (official) vs 0.072s (theoretical free-threaded)

## 🛠️ Development

### Adding New Tests
1. Add test methods to `tests/test_orchestrator.py`
2. Update configuration in `tests/test_config_*.yml`
3. Run tests with `uv run --python <version> python tests/test_orchestrator.py`

### Updating Visualizations
The visualizations automatically use real test data when available, falling back to configuration values when needed.

## 📈 Performance Insights

- Free-threaded Python shows significant threading performance improvements
- Scientific packages (numpy, scikit-learn) need ecosystem updates for compatibility
- uv provides seamless switching between Python versions
- Real-world performance gains depend on workload characteristics

## 🔧 Troubleshooting

### Package Compatibility Issues
- Free-threaded Python has limited package compatibility
- Use Python 3.14 official for production workloads
- Use free-threaded version for threading research

### Test Failures
- Check Python version compatibility
- Verify package installations
- Review test configuration files

## 📝 License

MIT License - see LICENSE file for details.