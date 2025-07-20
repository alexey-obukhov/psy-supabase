# 🚀 Psy-Supabase Startup Guide

This guide will help you start your psychological AI application with pain point monitoring on your production server.

## 📋 Prerequisites

### 1. System Requirements
- Python 3.8+
- NVIDIA GPU (optional, but recommended for better performance)
- 8GB+ RAM
- Self-hosted Supabase instance running at `192.168.2.150:8000`

### 2. Environment Setup

Create a `.env` file in your project root:

```bash
# Supabase Configuration
SUPABASE_URL=http://192.168.2.150:8000
SUPABASE_KEY=your_supabase_anon_key

# AI Model Configuration
TEXT_GENERATING_MODEL=microsoft/DialoGPT-medium
INTELLIGENT_PROCESS_ENABLED=true

# Flask Configuration
HOST=0.0.0.0
PORT=5008
FLASK_ENV=production

# Optional: CUDA Configuration
CUDA_VISIBLE_DEVICES=0
```

## 🔧 Installation

### Option 1: Using the Package (Recommended)

```bash
# Install the package with all dependencies
pip install -e .

# This will automatically install:
# - numpy>=2.0.0
# - torch>=2.4.1
# - transformers>=4.46.3
# - flask==3.0.3
# - supabase>=2.6.0
# - spacy==3.7.5
# - And all other dependencies
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for NLP processing)
python -m spacy download en_core_web_sm
```

## 🏃‍♂️ Starting the Application

### Method 1: Using the Package Entry Point (Recommended)

```bash
# Start the application using the installed console script
psy-supabase
```

### Method 2: Using Python Module

```bash
# Start as a module
python -m psy_supabase
```

### Method 3: Direct Python Execution

```bash
# Run the main file directly
python psy_supabase/main.py
```

### Method 4: Production Deployment with Gunicorn

```bash
# Install gunicorn for production
pip install gunicorn

# Start with gunicorn (production recommended)
gunicorn --bind 0.0.0.0:5008 --workers 2 --timeout 120 psy_supabase.main:app
```

## 🌐 Accessing Your Application

Once started, your application will be available at:

- **Local**: `http://localhost:5008`
- **Network**: `http://your-server-ip:5008`
- **Production**: `https://psy-supabase.com` (configure reverse proxy)

## 🔍 API Endpoints

### Core Endpoints

1. **Health Check**
   ```bash
   GET /health
   ```

2. **Chat with AI Therapist**
   ```bash
   POST /chat
   Headers: X-User-ID: your-user-id
   Body: {"question": "I'm feeling anxious about work"}
   ```

3. **Memory Status**
   ```bash
   GET /memory_status
   ```

### Pain Point Monitoring Endpoints

4. **Start Monitoring**
   ```bash
   POST /pain_point_monitoring/start
   Headers: X-User-ID: your-user-id
   ```

5. **Get Status**
   ```bash
   GET /pain_point_monitoring/status
   Headers: X-User-ID: your-user-id
   ```

6. **View Dashboard**
   ```bash
   GET /pain_point_monitoring/dashboard
   Headers: X-User-ID: your-user-id
   ```

7. **Stop Monitoring**
   ```bash
   POST /pain_point_monitoring/stop
   Headers: X-User-ID: your-user-id
   ```

## 🖥️ Web Interface

### Method 1: Using the Complete Webchat Interface

1. Open your browser and navigate to your deployed webchat:
   ```
   https://psy-supabase.com/webchat.html
   ```

2. The interface includes:
   - ✅ Full chat functionality
   - ✅ Pain point monitoring toggle
   - ✅ Real-time analytics
   - ✅ Live pain point detection
   - ✅ Floating monitoring widget

### Method 2: Testing Locally

If you want to test the HTML interface locally:

```bash
# Copy the webchat HTML to your web server directory
cp webchat_with_pain_point_monitoring.html /var/www/html/

# Or serve it locally with Python
python -m http.server 8080 --directory .
```

Then open: `http://localhost:8080/webchat_with_pain_point_monitoring.html`

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | Required | Your self-hosted Supabase URL |
| `SUPABASE_KEY` | Required | Supabase anonymous key |
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `5008` | Server port |
| `TEXT_GENERATING_MODEL` | `microsoft/DialoGPT-medium` | AI model to use |
| `INTELLIGENT_PROCESS_ENABLED` | `true` | Enable intelligent processing |

### GPU Configuration

If you have an NVIDIA GPU:

```bash
# Check GPU availability
nvidia-smi

# Set GPU for the application
export CUDA_VISIBLE_DEVICES=0
```

## 🚨 Pain Point Monitoring Features

Your application includes advanced pain point detection:

### Automatic Detection
- **Semantic Chunking**: Breaks down conversations into meaningful segments
- **Similarity Analysis**: Detects repetitive patterns in user messages
- **Theme Extraction**: Identifies psychological themes (anxiety, depression, etc.)
- **Intensity Assessment**: Classifies pain point severity

### Real-time Analytics
- **Live Dashboard**: Visual representation of conversation patterns
- **Pain Point Timeline**: Chronological view of detected issues
- **Theme Analysis**: Statistical breakdown of conversation topics
- **Therapeutic Recommendations**: AI-generated intervention suggestions

### Monitoring Controls
- **Toggle Monitoring**: Enable/disable pain point detection per session
- **Session Management**: Start/stop monitoring sessions
- **Export Data**: Access detailed analytics via API

## 🔍 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Install in development mode
   pip install -e .
   ```

2. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   ```

4. **Supabase Connection Issues**
   ```bash
   # Check your .env file and Supabase instance
   curl http://192.168.2.150:8000/health
   ```

### Logs and Debugging

Application logs are stored in:
- Console output (when running directly)
- `logging/` directory (file logs)

Enable debug mode:
```bash
export FLASK_ENV=development
```

## 🔐 Production Deployment

### Reverse Proxy Configuration (Nginx)

```nginx
server {
    listen 80;
    server_name psy-supabase.com;
    
    location / {
        proxy_pass http://127.0.0.1:5008;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

### SSL/HTTPS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d psy-supabase.com
```

### Systemd Service (Optional)

Create `/etc/systemd/system/psy-supabase.service`:

```ini
[Unit]
Description=Psy-Supabase AI Therapy Assistant
After=network.target

[Service]
Type=exec
User=your-user
WorkingDirectory=/path/to/psy-supabase
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/gunicorn --bind 127.0.0.1:5008 --workers 2 --timeout 120 psy_supabase.main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl enable psy-supabase
sudo systemctl start psy-supabase
sudo systemctl status psy-supabase
```

## 🎯 Quick Start Commands

For immediate testing:

```bash
# 1. Install and start
pip install -e .
psy-supabase

# 2. Test health check
curl http://localhost:5008/health

# 3. Test chat (replace USER_ID with actual ID)
curl -X POST http://localhost:5008/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test-user-123" \
  -d '{"question": "I am feeling anxious about my job"}'

# 4. Start pain point monitoring
curl -X POST http://localhost:5008/pain_point_monitoring/start \
  -H "X-User-ID: test-user-123"

# 5. Check monitoring status
curl http://localhost:5008/pain_point_monitoring/status \
  -H "X-User-ID: test-user-123"
```

## 📊 Expected Output

When everything is working correctly, you should see:

```
INFO - Application entry point executing...
INFO - Local development: Loading environment from .env file
INFO - Setting up application...
INFO - Welcome to the Therapy AI Assistant! Using model: microsoft/DialoGPT-medium on cuda
INFO - Model manager initialized for microsoft/DialoGPT-medium
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5008
 * Running on http://192.168.x.x:5008
```

Your psychological AI application with advanced pain point monitoring is now ready! 🎉

## 📞 Support

- Check logs in the `logging/` directory
- Monitor GPU usage with `nvidia-smi`
- Use `/memory_status` endpoint for memory monitoring
- Review pain point analytics via `/pain_point_monitoring/dashboard`
