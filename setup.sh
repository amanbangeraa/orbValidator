#!/bin/bash
# Setup script for ORB Seed Validation System

echo "🚁 Setting up IRoC-U 2026 ORB Seed Validation System"
echo "=================================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/seeds
mkdir -p data/queries

echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To run the system:"
echo "   1. Place reference images in data/seeds/"
echo "   2. Place test images in data/queries/"
echo "   3. Run: source venv/bin/activate && python3 run_test.py"
echo ""
echo "🔬 To run the demo:"
echo "   source venv/bin/activate && python3 demo.py"