#!/bin/bash
# Setup script for Free-Threaded Python Testing Suite
# This creates a virtual environment and installs all dependencies

echo "🚀 Setting up Free-Threaded Python Testing Environment"
echo "====================================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate and upgrade pip
echo "⬆️  Upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install numpy pandas scikit-learn matplotlib seaborn psutil

# Test installation
echo "🧪 Testing basic functionality..."
./venv/bin/python test_basic.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "💡 To use the virtual environment:"
echo "   source venv/bin/activate          # Activate environment"
echo "   python test_basic.py              # Basic test"
echo "   python visualize_results.py --simulate --save  # Generate plots"
echo "   deactivate                        # Deactivate when done"
echo ""
echo "🔧 Or use directly without activation:"
echo "   ./venv/bin/python test_basic.py   # Direct usage"
echo "   ./venv/bin/python run_tests.py --quick"
echo ""
echo "📁 Generated files will be in:"
echo "   example_plots/                    # Visualization outputs"
echo "   dataset_*.csv                     # Generated datasets"
echo "   *_results.json                    # Performance results"
