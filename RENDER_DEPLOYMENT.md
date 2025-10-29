# 🚀 Deploying EcoDrive Simulator on Render

Complete step-by-step guide for deploying the EcoDrive Simulator on Render.com.

## Overview

✅ **Yes! You can host BOTH frontend and backend on Render!**

Render offers two service types:
1. **Web Service** - For your Python/Flask backend
2. **Static Site** - For your React frontend

**This is the recommended setup** - hosting both on Render makes deployment simple and everything works seamlessly together.

### Quick Answer
- ✅ **Backend:** Deploy as **Web Service**
- ✅ **Frontend:** Deploy as **Static Site**
- ✅ **Both on Render** = Easy deployment, free tier available
- ✅ **Free tier:** Backend (spins down), Frontend (always on)
- ✅ **Paid tier:** $7/month for always-on backend

> 💡 **Quick Start:** See [RENDER_QUICK_GUIDE.md](./RENDER_QUICK_GUIDE.md) for fastest deployment steps.

---

## Option B: Separate Services (Recommended)

### Step 1: Deploy Backend Web Service

1. **Sign in to Render**
   - Go to https://render.com
   - Sign up or log in

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab repository
   - OR connect via GitHub manually

3. **Configure Backend Service**

   **Basic Settings:**
   - **Name:** `ecodrive-backend` (or your preferred name)
   - **Region:** Choose closest to your users
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty (or `backend/` if you prefer)
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     cd backend && pip install -r requirements.txt
     ```
     
     **Or if you need F1Tenth gym (for simulations):**
     ```bash
     cd f1tenth_rl-main && pip install -e . && cd .. && cd backend && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     cd backend && gunicorn --bind 0.0.0.0:$PORT --config gunicorn_config.py wsgi:app
     ```

   **Environment Variables:**
   Click "Add Environment Variable" and add:
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   SECRET_KEY=<generate-a-random-secret-key>
   CORS_ORIGINS=https://your-frontend-name.onrender.com,http://localhost:3000
   PORT=$PORT
   WORKERS=4
   LOG_LEVEL=info
   ```

   **Important Notes:**
   - `$PORT` is automatically set by Render - use this in your start command
   - Generate a secure `SECRET_KEY` (you can use: `python -c "import secrets; print(secrets.token_hex(32))"`)
   - Update `CORS_ORIGINS` after you deploy the frontend with the actual frontend URL

   **Advanced Settings:**
   - **Auto-Deploy:** Yes (deploys on git push)
   - **Docker:** No (using Python runtime)

4. **Add Disk (for Models)**
   - In "Advanced" section, add a disk if needed for model storage
   - Or use Render's persistent storage for `backend/models/` directory
   - **Note:** Model files are large - consider using external storage (S3, etc.)

5. **Click "Create Web Service"**
   - Render will start building your backend
   - Wait for deployment to complete (5-10 minutes)
   - Note the URL: `https://ecodrive-backend.onrender.com` (or your custom domain)

6. **Verify Backend is Running**
   - Visit: `https://your-backend-name.onrender.com/`
   - Should see: `{"message":"EcoDrive Simulator API","version":"1.0.0","status":"running"}`

---

### Step 2: Deploy Frontend Static Site

1. **Create New Static Site**
   - Click "New +" → "Static Site"
   - Connect the same repository

2. **Configure Frontend Service**

   **Basic Settings:**
   - **Name:** `ecodrive-frontend` (or your preferred name)
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty (root of repo)
   - **Build Command:**
     ```bash
     npm ci && REACT_APP_API_URL=https://your-backend-name.onrender.com npm run build
     ```
     
     > 💡 **Note:** Using `npm ci` instead of `npm install` for cleaner, more reliable builds
   - **Publish Directory:** `build`

   **Important:** Replace `https://your-backend-name.onrender.com` with your actual backend URL from Step 1!

   **Environment Variables:**
   Add these before building:
   ```
   REACT_APP_API_URL=https://your-backend-name.onrender.com
   REACT_APP_WS_URL=https://your-backend-name.onrender.com
   NODE_ENV=production
   ```

   **Advanced Settings:**
   - **Auto-Deploy:** Yes
   - **Pull Request Previews:** Optional

3. **Click "Create Static Site"**
   - Render will build and deploy your frontend
   - Frontend URL: `https://ecodrive-frontend.onrender.com`

4. **Update Backend CORS**
   - Go back to your backend service settings
   - Update `CORS_ORIGINS` environment variable:
     ```
     CORS_ORIGINS=https://ecodrive-frontend.onrender.com,https://your-backend-name.onrender.com
     ```
   - Save and redeploy backend

---

## Option A: Single Full-Stack Service (Alternative)

If you prefer to serve everything from one service:

1. **Create Web Service** (same as backend steps above)

2. **Update Start Command:**
   ```bash
   cd backend && gunicorn --bind 0.0.0.0:$PORT --config gunicorn_config.py wsgi:app
   ```

3. **Add Build Step to Serve Frontend:**
   Modify `backend/app.py` to serve static files (already handled if using Docker, but for Render you need to add):

   Add to `app.py`:
   ```python
   from flask import send_from_directory
   import os

   @app.route('/<path:path>')
   def serve_frontend(path):
       if path != "" and os.path.exists(os.path.join('../build', path)):
           return send_from_directory('../build', path)
       else:
           return send_from_directory('../build', 'index.html')

   @app.route('/')
   def serve_index():
       return send_from_directory('../build', 'index.html')
   ```

4. **Build Command:**
   ```bash
   npm install && REACT_APP_API_URL=https://your-service.onrender.com npm run build && cd backend && pip install -r requirements-prod.txt
   ```

---

## Model Storage on Render

### Option 1: Git Repository (Limited)
- Add models to `backend/models/` and commit to Git
- **Warning:** Large files (>100MB) may cause issues
- Not recommended for large model files

### Option 2: Render Disk (Recommended for Testing)
- Add persistent disk in Render settings
- Upload models via SSH or build script
- Limited to 100GB on free tier

### Option 3: External Storage (Recommended for Production)
- Store models on S3, Google Cloud Storage, or similar
- Modify `web_simulation.py` to download models on startup:
  ```python
  # Add to _run_simulation or __init__
  import boto3
  
  def download_models_from_s3(self):
      s3 = boto3.client('s3')
      s3.download_file('your-bucket', 'models/multi_track_ppo_2025-10-12_02-58-46.zip', 
                       'backend/models/multi_track_ppo_2025-10-12_02-58-46.zip')
  ```

### Option 4: Render Build Script
Create `render-build.sh`:
```bash
#!/bin/bash
# Download models during build
mkdir -p backend/models
# Use wget/curl to download from your storage
wget https://your-storage.com/models.zip -O backend/models/models.zip
unzip backend/models/models.zip -d backend/models/
```

Then in Render build command:
```bash
chmod +x render-build.sh && ./render-build.sh && cd backend && pip install -r requirements-prod.txt
```

---

## Environment Variables Reference

### Backend Service
```env
# Required
FLASK_ENV=production
PORT=$PORT
SECRET_KEY=<your-secret-key>

# CORS (update with your frontend URL)
CORS_ORIGINS=https://ecodrive-frontend.onrender.com

# Optional
WORKERS=4
LOG_LEVEL=info
MAX_SIMULATION_TIME=300.0
```

### Frontend Static Site
```env
# Required (set before building!)
REACT_APP_API_URL=https://your-backend-name.onrender.com
REACT_APP_WS_URL=https://your-backend-name.onrender.com

# Optional
NODE_ENV=production
```

---

## Render-Specific Considerations

### 1. Cold Starts
- Render free tier has "spin down" after 15 minutes of inactivity
- First request after spin down takes 30-60 seconds
- Upgrade to paid plan for always-on service

### 2. Build Time Limits
- Free tier: 15 minutes build time
- Large ML dependencies may take 10-15 minutes
- Upgrade if builds timeout

### 3. Memory Limits
- Free tier: 512MB RAM
- PyTorch/ML models need 1-2GB+ RAM
- **Recommendation:** Start with paid plan ($7/month) for 512MB, or upgrade to $25/month for 2.5GB

### 4. Disk Space
- Free tier: Limited temporary disk
- Models are large (100MB-1GB+ each)
- Use external storage (S3) for models

### 5. HTTPS
- Render provides free HTTPS certificates automatically
- URLs are: `https://your-service.onrender.com`
- Custom domains supported on paid plans

---

## Deployment Checklist

### Before First Deploy:
- [ ] Set `SECRET_KEY` environment variable
- [ ] Prepare model files (or setup external storage)
- [ ] Update CORS origins with your frontend URL
- [ ] Build and test locally first

### After Backend Deploy:
- [ ] Verify API endpoint works: `https://your-backend.onrender.com/`
- [ ] Test API endpoints manually
- [ ] Check logs for errors

### After Frontend Deploy:
- [ ] Verify frontend loads
- [ ] Check browser console for CORS errors
- [ ] Test API connection from frontend
- [ ] Update backend CORS if needed

### Post-Deployment:
- [ ] Monitor Render logs
- [ ] Set up error alerting (optional)
- [ ] Test full simulation flow
- [ ] Monitor resource usage

---

## Troubleshooting

### Build Failures

**"Build timeout"**
- Upgrade to paid plan for longer build times
- Optimize requirements (remove unused packages)
- Use build cache

**"Out of memory during build"**
- Build locally and commit `node_modules` (not recommended)
- Or upgrade Render plan

**"Module not found"**
- Check `requirements.txt` includes all dependencies
- Verify build command installs requirements
- F1Tenth gym must be installed separately if you want simulations to work

### Runtime Issues

**"502 Bad Gateway"**
- Check Render logs
- Verify start command is correct
- Check if service spun down (free tier)

**"CORS errors"**
- Verify `CORS_ORIGINS` includes your frontend URL
- Check frontend `REACT_APP_API_URL` matches backend URL
- Ensure URLs use `https://` not `http://`

**"Models not found"**
- Verify models directory exists
- Check model paths in `web_simulation.py`
- Ensure models are in Git or external storage

**"Cold start too slow"**
- Upgrade to paid plan for always-on service
- Or pre-warm service with cron job

### Logs Access
- Go to your service → "Logs" tab
- Check both build logs and runtime logs
- Look for Python errors, import errors, etc.

---

## Cost Estimation

### Free Tier
- **Backend:** Free (spins down after 15 min inactivity)
- **Frontend:** Free (always on)
- **Total:** $0/month

### Paid Tier (Recommended)
- **Backend:** $7/month (512MB RAM, always-on)
- **Frontend:** Free
- **Total:** ~$7/month

### Production Tier
- **Backend:** $25/month (2.5GB RAM, always-on, better performance)
- **Frontend:** Free
- **Total:** ~$25/month

**Note:** Storage for models not included - use S3 or similar ($0.023/GB/month).

---

## Next Steps

1. **Deploy Backend** following Step 1 above
2. **Deploy Frontend** following Step 2 above
3. **Test** both services
4. **Update CORS** if needed
5. **Monitor** logs and performance
6. **Upgrade** plan if needed for production

---

## Support Resources

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Render Status: https://status.render.com

---

**Ready to deploy? Start with Step 1: Deploy Backend Web Service! 🚀**

