# 📦 Requirements Files Explained

## Overview

There are **two requirements files** for different use cases:

### 1. `backend/requirements.txt` 
**Use for:** Both development and production
- Contains ALL dependencies including Gunicorn
- Good for: Local development, production deployment
- **This is the one to use for Render**

### 2. `backend/requirements-prod.txt`
**Use for:** Production only (alternative)
- Same dependencies as requirements.txt
- Organized with comments
- Alternative if you want a separate production file

## For Render Deployment

**Use `requirements.txt`** - It has everything needed including Gunicorn.

In your Render build command:
```bash
cd backend && pip install -r requirements.txt
```

## Installation Notes

### F1Tenth Gym Package

The `f1tenth_gym` package is **NOT in requirements.txt** because it's a local package that must be installed separately:

1. **Install from local directory:**
   ```bash
   cd f1tenth_rl-main
   pip install -e .
   ```
   
   OR

2. **Install from the wrapper path:**
   ```bash
   cd f1tenth_rl-main/src
   pip install -e ./f1tenth_wrapper  # If this is a package
   ```

The code gracefully handles missing F1Tenth dependencies (see `web_simulation.py` lines 26-62), so the app will work without it but simulations won't run.

### For Render Deployment

Since Render builds from your repository, you have two options:

#### Option 1: Install F1Tenth during Render build
Add to your Render build command:
```bash
cd f1tenth_rl-main && pip install -e . && cd .. && cd backend && pip install -r requirements.txt
```

#### Option 2: Make F1Tenth optional (current setup)
The app already handles missing F1Tenth gracefully - simulations just won't work. You can:
1. Deploy without F1Tenth (API will work, but no simulations)
2. Add F1Tenth installation to build command later
3. Or create a proper Python package for f1tenth_gym

## What's Included

### Core Dependencies ✅
- Flask, Flask-CORS, Flask-SocketIO
- Gunicorn (production WSGI server)
- Gevent (async support for Flask-SocketIO)

### ML/AI Dependencies ✅
- PyTorch (torch)
- Stable-Baselines3 (PPO, SAC)
- Gymnasium (RL environments)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### F1Tenth Dependencies ✅
- OpenCV (opencv-python)
- Pillow (PIL)
- Numba, SciPy

### Data Dependencies ✅
- FastF1 (F1 API)
- Pandas, NumPy

### System Dependencies ✅
- psutil (system monitoring)
- python-dotenv (environment variables)

## Missing Dependencies

### F1Tenth Gym ⚠️
- **Status:** Must be installed separately from `f1tenth_rl-main/`
- **Impact:** Simulations won't work without it (API will still function)
- **Solution:** Install during Render build or make it optional

## Testing Requirements

To verify everything is installed:
```bash
cd backend
pip install -r requirements.txt

# Test imports
python -c "import flask; import torch; import stable_baselines3; print('Core dependencies OK')"
python -c "import f1tenth_gym; print('F1Tenth OK')"  # Optional
```

## For Local Development

```bash
# Install all requirements
cd backend
pip install -r requirements.txt

# Install F1Tenth gym (if you want simulations)
cd ../f1tenth_rl-main
pip install -e .
```

## Summary

✅ **`requirements.txt`** = **Complete** for everything except F1Tenth  
⚠️ **F1Tenth gym** = Must install separately from local directory  
✅ **For Render** = Use `requirements.txt` + optional F1Tenth installation in build command

