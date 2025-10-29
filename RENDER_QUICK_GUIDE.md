# 🚀 Quick Render Deployment Guide

## ✅ Yes! Host Both Frontend & Backend on Render

Render is perfect for hosting both parts of your application:

- **Backend** → Web Service (Python/Flask)
- **Frontend** → Static Site (React)

---

## Step-by-Step Deployment

### 🎯 Step 1: Deploy Backend (Web Service)

1. Go to https://render.com and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:

   **Service Name:** `ecodrive-backend`
   
   **Settings:**
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     cd backend && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     cd backend && gunicorn --bind 0.0.0.0:$PORT --config gunicorn_config.py wsgi:app
     ```

   **Environment Variables:**
   ```
   FLASK_ENV=production
   PORT=$PORT
   SECRET_KEY=<generate-random-key>
   CORS_ORIGINS=https://your-frontend-name.onrender.com
   ```

5. Click **"Create Web Service"**
6. Wait for deployment (~5-10 minutes)
7. **Note your backend URL:** `https://ecodrive-backend.onrender.com`

---

### 🎯 Step 2: Deploy Frontend (Static Site)

1. In Render dashboard, click **"New +"** → **"Static Site"**
2. Connect the same GitHub repository
3. Configure:

   **Service Name:** `ecodrive-frontend`
   
   **Settings:**
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty
   - **Build Command:**
     ```bash
     npm ci && npm run build
     ```
     
     **Or with environment variable:**
     ```bash
     npm ci && REACT_APP_API_URL=https://ecodrive-backend.onrender.com npm run build
     ```
     
     > 💡 **Note:** If you get permission errors, use `npm ci` instead of `npm install`
     ⚠️ **Important:** Replace `ecodrive-backend.onrender.com` with your actual backend URL!
   
   - **Publish Directory:** `build`
   
   **Environment Variables:**
   ```
   REACT_APP_API_URL=https://ecodrive-backend.onrender.com
   REACT_APP_WS_URL=https://ecodrive-backend.onrender.com
   ```
   ⚠️ **Important:** Set these BEFORE your first build!

4. Click **"Create Static Site"**
5. Wait for deployment (~3-5 minutes)
6. **Note your frontend URL:** `https://ecodrive-frontend.onrender.com`

---

### 🎯 Step 3: Update CORS (Important!)

1. Go back to your **Backend Web Service**
2. Open **"Environment"** tab
3. Update `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=https://ecodrive-frontend.onrender.com
   ```
   (Replace with your actual frontend URL)
4. Save - Render will automatically redeploy

---

### 🎯 Step 4: Test

1. Visit your frontend URL: `https://ecodrive-frontend.onrender.com`
2. Check browser console for errors
3. Try starting a simulation
4. Check backend logs in Render dashboard if issues occur

---

## 💰 Pricing

### Free Tier
- ✅ **Backend:** Free (spins down after 15 min inactivity)
- ✅ **Frontend:** Free (always on)
- **Total:** $0/month

### Paid Tier (Recommended)
- **Backend:** $7/month (always-on, 512MB RAM)
- **Frontend:** Free
- **Total:** ~$7/month

---

## 🔑 Important Notes

### Backend URL
- Get your backend URL after Step 1
- Use this URL in frontend environment variables

### CORS Configuration
- Must match your frontend URL exactly
- Use `https://` not `http://`
- No trailing slashes

### Environment Variables
- Set them BEFORE first build
- Frontend variables are "baked in" during build
- If you change them, you must rebuild

### Cold Starts (Free Tier)
- Backend sleeps after 15 min inactivity
- First request takes 30-60 seconds to wake up
- Upgrade to paid plan for always-on

---

## 🐛 Troubleshooting

**"CORS error"**
- Check `CORS_ORIGINS` includes frontend URL
- Verify URLs use `https://`
- Rebuild frontend if you changed environment variables

**"Backend not responding"**
- Check if backend spun down (free tier)
- Wait 30-60 seconds for cold start
- Check Render logs

**"404 on frontend"**
- Verify build completed successfully
- Check `Publish Directory` is set to `build`
- Check Render build logs

---

## ✅ Success Checklist

- [ ] Backend deployed and accessible
- [ ] Frontend deployed and accessible  
- [ ] CORS configured correctly
- [ ] Environment variables set
- [ ] Frontend can connect to backend
- [ ] No console errors
- [ ] Simulation starts correctly

---

**That's it! Both frontend and backend are now on Render! 🎉**

For detailed instructions, see [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

