"""
Gunicorn configuration for production deployment
"""
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
# Start with 1 worker to avoid timeout during heavy imports (torch, stable-baselines3, etc.)
# Increase via WORKERS env var once deployed
workers = int(os.environ.get('WORKERS', 1))
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Increased for heavy ML model imports
keepalive = 5
graceful_timeout = 300

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "ecodrive-simulator"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Worker lifecycle - preload app so heavy ML libraries (torch, etc.) load once
# Workers fork after app is loaded, avoiding repeated imports
preload_app = True

# Worker limits to prevent memory issues
max_requests = 1000
max_requests_jitter = 50

# SSL (if needed)
# keyfile = None
# certfile = None

