#!/bin/bash

# 🚀 Psy-Supabase Quick Start Script
# This script helps you start your psychological AI application

set -e  # Exit on any error

echo "🧠 Psy-Supabase Startup Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "psy_supabase" ]; then
    print_error "Please run this script from the psy-supabase project root directory"
    exit 1
fi

print_info "Checking system requirements..."

# Check Python version
if ! python3 --version &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python version: $python_version"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating a template..."
    cat > .env << EOF
# Supabase Configuration (REQUIRED)
SUPABASE_URL=http://192.168.2.150:8000
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicmVmIjoibG9jYWxob3N0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI4Njc0NTYsImV4cCI6MjA2ODIyNzQ1Nn0.OilBLadMfjrrqFSofdDOKlin0j5p4qpyu6fF_ZqFeaw

# AI Model Configuration
TEXT_GENERATING_MODEL=rasyosef/Phi-1_5-Instruct-v0.1
INTELLIGENT_PROCESS_ENABLED=true

# Flask Configuration
HOST=0.0.0.0
PORT=5000
FLASK_ENV=production

# Optional: CUDA Configuration
CUDA_VISIBLE_DEVICES=0
EOF
    print_warning "Please edit .env file with your actual Supabase credentials before starting"
    echo
fi

# Check if virtual environment should be created
if [ "$1" = "--create-venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_status "Virtual environment created and activated"
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Not running in a virtual environment. Consider using: source venv/bin/activate"
fi

# Install or upgrade the package
print_info "Installing/upgrading psy-supabase package..."
pip install -e . --upgrade

# Check if spaCy model is available
print_info "Checking spaCy model..."
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    print_info "Downloading spaCy English model..."
    python3 -m spacy download en_core_web_sm
    print_status "spaCy model downloaded"
else
    print_status "spaCy model already available"
fi

# Check GPU availability
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
    print_status "GPU available: $gpu_name"
else
    print_warning "No GPU detected. Running on CPU (slower performance)"
fi

# Check Supabase connection
print_info "Checking Supabase connection..."
source .env 2>/dev/null || true
if [ -n "$SUPABASE_URL" ] && [ "$SUPABASE_URL" != "your_supabase_url_here" ]; then
    if curl -s "$SUPABASE_URL/health" &> /dev/null; then
        print_status "Supabase connection successful"
    else
        print_warning "Cannot connect to Supabase at $SUPABASE_URL"
        print_warning "Make sure your Supabase instance is running"
    fi
else
    print_warning "SUPABASE_URL not configured in .env file"
fi

echo
print_info "System check complete! Starting the application..."
echo

# Determine startup method
if [ "$1" = "--gunicorn" ]; then
    print_info "Starting with Gunicorn (production mode)..."
    if ! command -v gunicorn &> /dev/null; then
        print_info "Installing gunicorn..."
        pip install gunicorn
    fi
    exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --access-logfile - psy_supabase.main:app
elif [ "$1" = "--debug" ]; then
    print_info "Starting in debug mode..."
    export FLASK_ENV=development
    exec python3 -m psy_supabase
else
    print_info "Starting in normal mode..."
    print_info "Available at: http://localhost:5000"
    print_info "Health check: http://localhost:5000/health"
    print_info "Webchat: Copy webchat_with_pain_point_monitoring.html to your web server"
    echo
    print_info "Use Ctrl+C to stop the server"
    echo
    exec psy-supabase
fi
