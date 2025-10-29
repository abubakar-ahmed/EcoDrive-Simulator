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

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/build ./static

# Copy models directory (if exists)
COPY backend/models ./backend/models

# Set environment variables
ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run Gunicorn server
# Change to backend directory to match Procfile behavior where imports are relative
WORKDIR /app/backend
CMD ["gunicorn", "--config", "gunicorn_config.py", "wsgi:app"]

