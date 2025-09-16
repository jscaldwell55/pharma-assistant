# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes
# Use only 1 worker to minimize memory usage
workers = 1
worker_class = 'sync'
worker_connections = 1000
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 20

# Timeout settings
timeout = 120  # Increased to 120 seconds for model loading
graceful_timeout = 30
keepalive = 2

# Development settings
reload = False
spew = False
check_config = False

# Server mechanics
daemon = False
raw_env = []
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
errorlog = '-'
loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'pharma-assistant'

# Server hooks
def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info("Pre-fork: preparing to fork worker")

def post_fork(server, worker):
    """Called just after a worker is forked"""
    server.log.info(f"Post-fork: worker {worker.pid} forked")

def pre_exec(server):
    """Called just before a new master process is forked"""
    server.log.info("Pre-exec: forking new master process")

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Server is ready. Spawning workers")

def worker_init(worker):
    """Called just after a worker has initialized the application"""
    worker.log.info(f"Worker {worker.pid} initialized")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT"""
    worker.log.info(f"Worker {worker.pid} interrupted")

def on_exit(server):
    """Called just before exiting"""
    server.log.info("Server shutting down")

# Preload the application to share memory between workers
# But with only 1 worker, this is less critical
preload_app = True

# StatsD (optional monitoring)
statsd_host = None
statsd_port = 8125
statsd_prefix = 'pharma_assistant'