# 🔧 Fix: Permission Denied Error on Render

## Problem
```
sh: 1: react-scripts: Permission denied
==> Build failed 😞
```

## Solution

This is a permissions issue with node_modules. Use one of these fixes:

### Fix 1: Use `npm ci` (Recommended)

Update your **Build Command** in Render Static Site settings:

```bash
npm ci && REACT_APP_API_URL=https://your-backend-url.onrender.com npm run build
```

**Why `npm ci`?**
- Clean install (removes node_modules first)
- Faster and more reliable
- Ensures proper permissions
- Uses package-lock.json exactly

### Fix 2: Use `npx` to run react-scripts

Update your **Build Command**:

```bash
npm install && REACT_APP_API_URL=https://your-backend-url.onrender.com npx react-scripts build
```

### Fix 3: Fix permissions explicitly

Update your **Build Command**:

```bash
npm install && chmod +x node_modules/.bin/* && REACT_APP_API_URL=https://your-backend-url.onrender.com npm run build
```

### Fix 4: Clean rebuild

Update your **Build Command**:

```bash
rm -rf node_modules package-lock.json && npm install && REACT_APP_API_URL=https://your-backend-url.onrender.com npm run build
```

## Recommended Solution

**Use Fix 1** (`npm ci`) - it's the cleanest and most reliable:

```bash
npm ci && REACT_APP_API_URL=https://ecodrive-backend.onrender.com npm run build
```

Replace `ecodrive-backend.onrender.com` with your actual backend URL.

## Environment Variables Alternative

Instead of putting the URL in the build command, you can set it as an environment variable in Render:

**Environment Variables (in Render dashboard):**
```
REACT_APP_API_URL=https://your-backend-url.onrender.com
REACT_APP_WS_URL=https://your-backend-url.onrender.com
```

**Then Build Command becomes:**
```bash
npm ci && npm run build
```

This is cleaner and easier to update!

## Steps to Fix

1. Go to your Static Site in Render dashboard
2. Click **"Manual Deploy"** or **"Settings"**
3. Update **Build Command** to use `npm ci`:
   ```bash
   npm ci && REACT_APP_API_URL=https://your-backend-url.onrender.com npm run build
   ```
4. **OR** add environment variables instead:
   - Go to **Environment** tab
   - Add `REACT_APP_API_URL` and `REACT_APP_WS_URL`
   - Use build command: `npm ci && npm run build`
5. Save and redeploy

---

**The `npm ci` fix should resolve the permission issue!** ✅

