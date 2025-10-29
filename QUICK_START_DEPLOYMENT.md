# 🚀 Quick Start Deployment Guide

## Option 1: Docker (Easiest)

### Prerequisites
- Docker and Docker Compose installed

### Steps
1. **Build and run:**
   ```bash
   docker-compose up -d
   ```

2. **Access the application:**
   - Frontend + Backend: http://localhost:5000

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

## Option 2: Manual Deployment

### Backend Setup
```bash
cd backend
pip install -r requirements-prod.txt
python wsgi.py
```

### Frontend Setup
```bash
# Terminal 1: Build frontend
npm install
npm run build

# Terminal 2: Serve (optional - or deploy build/ folder)
# Use nginx, apache, or any static file server
```

## Option 3: Cloud Platforms

### Railway / Render / Fly.io

1. **Connect your GitHub repository**

2. **Set environment variables:**
   ```
   PORT=5000
   FLASK_ENV=production
   CORS_ORIGINS=https://your-frontend-domain.com
   REACT_APP_API_URL=https://your-backend-domain.com
   ```

3. **Build commands:**
   - Backend: `cd backend && pip install -r requirements-prod.txt`
   - Start: `cd backend && gunicorn --config gunicorn_config.py wsgi:app`

4. **Frontend:** Deploy `build/` folder separately or use static hosting

## Environment Variables

### Backend (.env in backend/)
```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
```

### Frontend (.env in root/)
```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=http://localhost:5000
```

**Note:** For production, set `REACT_APP_API_URL` to your backend URL before building:
```bash
REACT_APP_API_URL=https://api.yourdomain.com npm run build
```

## Troubleshooting

- **Port already in use:** Change `FLASK_PORT` in `.env`
- **CORS errors:** Add frontend URL to `CORS_ORIGINS`
- **Models not found:** Ensure `backend/models/` directory exists with trained models

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md)

