# 🚀 EcoDrive Simulator Deployment Guide

Complete guide for deploying the EcoDrive Simulator to various hosting platforms.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Environment Configuration](#environment-configuration)
4. [Platform-Specific Guides](#platform-specific-guides)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for containerized deployment)
- Trained ML models in `backend/models/`

## Deployment Options

### Option 1: Docker Deployment (Recommended)
- Simplest deployment
- Consistent across environments
- Includes both frontend and backend

### Option 2: Separate Frontend/Backend
- Frontend: Static hosting (Vercel, Netlify, Cloudflare Pages)
- Backend: Cloud platform (Heroku, Railway, Render, AWS, GCP, Azure)

### Option 3: Full-Stack Platform
- Platforms like Render, Railway, Fly.io support both

## Environment Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://your-frontend-domain.com
MAX_SIMULATION_TIME=300.0
LOG_LEVEL=INFO
WORKERS=4
```

### Frontend Environment Variables

Create a `.env` file in the root directory:

```env
REACT_APP_API_URL=https://your-backend-domain.com
REACT_APP_WS_URL=https://your-backend-domain.com
```

## Platform-Specific Guides

### 1. Docker Deployment

#### Build and Run Locally

```bash
# Build the Docker image
docker build -t ecodrive-simulator .

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -p 5000:5000 -e PORT=5000 ecodrive-simulator
```

#### Deploy to Cloud Platforms

**AWS ECS / Google Cloud Run / Azure Container Instances:**
- Push Docker image to container registry
- Deploy using platform's container service
- Set environment variables in platform config

### 2. Railway.app

#### Backend Deployment

1. Create new project on Railway
2. Connect GitHub repository
3. Set root directory to project root
4. Configure build settings:
   - Build command: `cd backend && pip install -r requirements-prod.txt`
   - Start command: `cd backend && gunicorn --config gunicorn_config.py wsgi:app`

5. Set environment variables in Railway dashboard:
   ```
   PORT=5000
   FLASK_ENV=production
   CORS_ORIGINS=https://your-frontend.railway.app
   ```

#### Frontend Deployment

1. Build frontend locally or use Railway:
   ```bash
   npm run build
   ```
2. Deploy `build/` folder to static hosting or Railway static site

### 3. Render.com

#### Backend Service

1. Create new Web Service
2. Connect repository
3. Build settings:
   - Build Command: `cd backend && pip install -r requirements-prod.txt`
   - Start Command: `cd backend && gunicorn --bind 0.0.0.0:$PORT --config gunicorn_config.py wsgi:app`

4. Environment variables:
   ```
   PORT=$PORT (automatically set by Render)
   FLASK_ENV=production
   CORS_ORIGINS=https://your-frontend.onrender.com
   ```

#### Frontend Static Site

1. Create new Static Site
2. Build command: `npm install && npm run build`
3. Publish directory: `build`

### 4. Heroku

#### Backend

1. Install Heroku CLI
2. Create `Procfile` in project root:
   ```
   web: cd backend && gunicorn --bind 0.0.0.0:$PORT --config gunicorn_config.py wsgi:app
   ```

3. Create `runtime.txt`:
   ```
   python-3.11.6
   ```

4. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set FLASK_ENV=production
   heroku config:set CORS_ORIGINS=https://your-frontend.herokuapp.com
   git push heroku main
   ```

#### Frontend

1. Build frontend:
   ```bash
   npm run build
   ```

2. Deploy `build/` folder using Heroku static buildpack or separate hosting

### 5. AWS (Elastic Beanstalk / EC2)

#### Elastic Beanstalk

1. Create `.ebextensions/python.config`:
   ```yaml
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: wsgi:app
   ```

2. Deploy:
   ```bash
   eb init
   eb create ecodrive-simulator
   eb deploy
   ```

#### EC2 with Docker

1. Launch EC2 instance
2. Install Docker
3. Build and run:
   ```bash
   docker build -t ecodrive-simulator .
   docker run -d -p 80:5000 ecodrive-simulator
   ```

### 6. Google Cloud Platform

#### Cloud Run

1. Build and push:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/ecodrive-simulator
   gcloud run deploy ecodrive-simulator \
     --image gcr.io/PROJECT-ID/ecodrive-simulator \
     --platform managed \
     --port 5000
   ```

#### App Engine

1. Create `app.yaml`:
   ```yaml
   runtime: python311
   entrypoint: gunicorn --bind 0.0.0.0:$PORT --config backend/gunicorn_config.py wsgi:app
   env_variables:
     FLASK_ENV: production
   ```

2. Deploy:
   ```bash
   gcloud app deploy
   ```

### 7. DigitalOcean

#### App Platform

1. Create new app from GitHub
2. Set build and run commands (similar to Render)
3. Configure environment variables

#### Droplet with Docker

1. Create droplet
2. Install Docker
3. Use docker-compose or docker run

### 8. Fly.io

1. Install Fly CLI:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. Create `fly.toml`:
   ```toml
   app = "ecodrive-simulator"
   primary_region = "iad"

   [build]
     dockerfile = "Dockerfile"

   [[services]]
     internal_port = 5000
     protocol = "tcp"
   ```

3. Deploy:
   ```bash
   fly deploy
   ```

## Frontend Updates Required

After deployment, update frontend API URLs. Update these files to use the config:

1. **SimulationDashboard.js** - Replace `http://localhost:5000` with `config.getApiUrl(config.endpoints.simulation.status)`
2. **SimulationSetup.js** - Replace hardcoded URLs
3. **ResultsComparison.js** - Replace hardcoded URLs
4. **TelemetryGraph.js** - Replace hardcoded URLs

Or set `REACT_APP_API_URL` environment variable before building.

## Production Checklist

- [ ] Set `FLASK_DEBUG=False`
- [ ] Generate strong `SECRET_KEY`
- [ ] Configure `CORS_ORIGINS` with your frontend domain
- [ ] Set proper `WORKERS` count (CPU count * 2 + 1)
- [ ] Ensure models directory is accessible
- [ ] Set up SSL/HTTPS certificates
- [ ] Configure logging and monitoring
- [ ] Set up backup for models and cache
- [ ] Test API endpoints after deployment
- [ ] Monitor resource usage (CPU, memory, disk)

## Troubleshooting

### Backend won't start
- Check port is available: `netstat -an | grep 5000`
- Verify Python dependencies installed
- Check logs: `docker logs <container>` or platform logs

### CORS errors
- Ensure `CORS_ORIGINS` includes your frontend URL
- Check backend CORS configuration in `app.py`

### Models not found
- Verify `backend/models/` directory exists
- Check model paths in `web_simulation.py`
- Ensure models are included in deployment

### Performance issues
- Increase worker count
- Use GPU if available for ML inference
- Optimize model loading (cache models)
- Consider model quantization for faster inference

### Out of memory
- Reduce `WORKERS` count
- Use smaller models or model quantization
- Increase server memory
- Optimize image rendering (reduce frame rate)

## Monitoring & Maintenance

### Health Checks

The backend includes health check endpoint:
```
GET /api/health
```

### Logging

Logs are output to stdout/stderr. For production:
- Use log aggregation service (Datadog, Loggly, etc.)
- Set `LOG_LEVEL=INFO` or `WARNING` for production

### Updates

1. Pull latest code
2. Rebuild Docker image or redeploy
3. Test on staging first
4. Monitor after deployment

## Support

For deployment issues:
- Check platform-specific documentation
- Review logs carefully
- Test locally with Docker first
- Verify all environment variables are set

