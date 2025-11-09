# 🔧 Health Check Configuration Guide

## ✅ Current Configuration Status

Your `render.yaml` is **correctly configured**:
- **FastAPI Service**: Health path = `/health` ✅
- **Streamlit Service**: Health path = `/_stcore/health` ✅

## 🚨 If Using Railway Instead of Render

If you're deploying to **Railway**, you need to configure it manually:

### For FastAPI Service (Dockerfile)

**Railway Settings:**
1. Go to your service → **Settings** → **Health Check**
2. Set **Health Check Path**: `/health`
3. Set **Port**: `8000`
4. Add Environment Variable: `PORT=8000`

**Verify Dockerfile:**
```dockerfile
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### For Streamlit Service (Dockerfile.streamlit)

**Railway Settings:**
1. Go to your service → **Settings** → **Health Check**
2. Set **Health Check Path**: `/_stcore/health`
3. Set **Port**: `8501`
4. Add Environment Variable: `PORT=8501`

**Verify Dockerfile.streamlit:**
```dockerfile
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

## 🔍 Quick Diagnosis

### Check 1: Which Service Are You Deploying?

**If deploying FastAPI:**
- ✅ Health path must be: `/health`
- ✅ Port must be: `8000`
- ✅ Dockerfile: `Dockerfile`

**If deploying Streamlit:**
- ✅ Health path must be: `/_stcore/health`
- ✅ Port must be: `8501`
- ✅ Dockerfile: `Dockerfile.streamlit`

### Check 2: Verify Your Dockerfile CMD

**FastAPI (Dockerfile):**
```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Streamlit (Dockerfile.streamlit):**
```dockerfile
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Check 3: Check Logs

**For FastAPI, you should see:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**For Streamlit, you should see:**
```
You can now view your Streamlit app in your browser.
Network URL: http://0.0.0.0:8501
```

## 🛠️ Fix Steps for Railway

### Step 1: Identify Which Service Failed

Check your Railway logs to see which service is failing.

### Step 2: Configure Health Check

**For FastAPI:**
1. Railway Dashboard → Your Service → **Settings**
2. **Health Check Path**: Change to `/health`
3. **Port**: Set to `8000`
4. **Environment Variables**: Add `PORT=8000`

**For Streamlit:**
1. Railway Dashboard → Your Service → **Settings**
2. **Health Check Path**: Change to `/_stcore/health`
3. **Port**: Set to `8501`
4. **Environment Variables**: Add `PORT=8501`

### Step 3: Redeploy

After changing settings, Railway will automatically redeploy.

## ✅ Verification

### Test Health Endpoints Locally

**FastAPI:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","model_loaded":true}
```

**Streamlit:**
```bash
curl http://localhost:8501/_stcore/health
# Should return: {"status":"ok"}
```

### Test in Production

**FastAPI:**
```
https://your-api.onrender.com/health
# or
https://your-api.railway.app/health
```

**Streamlit:**
```
https://your-streamlit.onrender.com/_stcore/health
# or
https://your-streamlit.railway.app/_stcore/health
```

## 📋 Summary Table

| Service | Dockerfile | Port | Health Path | CMD |
|---------|-----------|------|-------------|-----|
| FastAPI | `Dockerfile` | 8000 | `/health` | `uvicorn app:app --host 0.0.0.0 --port 8000` |
| Streamlit | `Dockerfile.streamlit` | 8501 | `/_stcore/health` | `streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0` |

## 🎯 Current Status

✅ **Your Dockerfiles are correct**
✅ **Your render.yaml is correct**
✅ **streamlit_app.py has been recreated**

If you're using **Render**, the configuration should work automatically.

If you're using **Railway**, follow the manual configuration steps above.

