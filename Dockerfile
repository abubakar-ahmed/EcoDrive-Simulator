# Multi-stage Dockerfile for EcoDrive Simulator
# Build frontend and serve with Python backend

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY src ./src
COPY public ./public

# Build React app
RUN npm run build

# Stage 2: Python backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend code
COPY backend ./backend

# Copy f1tenth_rl-main directories needed for simulation
# Copy f1tenth_gym and src directories for f1tenth_wrapper
# Ensure directory structure exists before copying
RUN mkdir -p /app/f1tenth_rl-main
COPY f1tenth_rl-main/f1tenth_gym /app/f1tenth_rl-main/f1tenth_gym
COPY f1tenth_rl-main/src /app/f1tenth_rl-main/src
# Create maps directory that will be symlinked to backend/maps at runtime
# This allows f1tenth_gym to find maps at /app/f1tenth_rl-main/maps
RUN mkdir -p /app/f1tenth_rl-main/maps
# Verify the directories were copied correctly and test Python import
RUN echo "=== Verifying f1tenth_rl-main directories ===" && \
    ls -la /app/f1tenth_rl-main/ && \
    echo "--- src directory ---" && \
    ls -la /app/f1tenth_rl-main/src/ && \
    echo "--- f1tenth_gym directory ---" && \
    ls -la /app/f1tenth_rl-main/f1tenth_gym/ && \
    echo "--- f1tenth_wrapper directory ---" && \
    ls -la /app/f1tenth_rl-main/src/f1tenth_wrapper/ && \
    echo "--- Testing Python import paths ---" && \
    python3 -c "import sys; sys.path.insert(0, '/app/f1tenth_rl-main/src'); sys.path.insert(0, '/app/f1tenth_rl-main/f1tenth_gym'); import f1tenth_wrapper; print('✅ f1tenth_wrapper imported successfully')" && \
    echo "✅ f1tenth_rl-main directories verified"

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/build ./static

# Copy models directory (if exists)
COPY backend/models ./backend/models

# Set environment variables
ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
# Add f1tenth_rl-main paths to PYTHONPATH so Python can find f1tenth_gym and f1tenth_wrapper
# src must be in PYTHONPATH so f1tenth_wrapper (under src/) can be imported
# f1tenth_gym must be in PYTHONPATH so f1tenth_gym can be imported
ENV PYTHONPATH=/app/f1tenth_rl-main/src:/app/f1tenth_rl-main/f1tenth_gym:/app:${PYTHONPATH}
# Verify PYTHONPATH is set (for debugging)
RUN echo "PYTHONPATH will be: $PYTHONPATH" && \
    python3 -c "import os; print('PYTHONPATH at build time:', os.environ.get('PYTHONPATH', 'NOT SET'))"

# Expose port
EXPOSE 5000

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run Gunicorn server
# Change to backend directory to match Procfile behavior where imports are relative
WORKDIR /app/backend
CMD ["gunicorn", "--config", "gunicorn_config.py", "wsgi:app"]

