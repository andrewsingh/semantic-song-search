# Deployment Guide - Semantic Song Search

## Quick Deploy to Railway (Recommended)

Railway is the easiest modern alternative to Heroku for this app.

### Prerequisites
- GitHub account with your code pushed
- Spotify Developer App with Client ID and Secret
- OpenAI API key
- (Optional) Mixpanel token for analytics

### Step-by-Step Railway Deployment

1. **Sign up at [railway.app](https://railway.app)** and connect your GitHub

2. **Create a new project** from your GitHub repository

3. **Set Environment Variables** in Railway dashboard:
   ```
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   OPENAI_API_KEY=your_openai_api_key
   MIXPANEL_TOKEN=your_mixpanel_token (optional)
   FLASK_SECRET_KEY=your_random_secret_key
   ```

4. **Upload Data Files**: 
   - In Railway dashboard, go to your service
   - Navigate to "Volumes" and create a new volume
   - Upload your `data/` directory (or use Railway's file upload)
   - Alternative: The files are included in your git repo and will deploy automatically

5. **Deploy**: Railway will automatically build and deploy your app

6. **Get your URL**: Railway provides a custom domain like `your-app-name.railway.app`

### Data Files Handling

Your app needs these files (~350MB total):
- `data/eval_set_v2/eval_set_v2_metadata_ready.json` (37MB)
- `data/eval_set_v2/eval_set_v2_embeddings.npz` (319MB) OR
- `data/eval_set_v2/eval_set_v2_embeddings/` directory with separate files

**Options:**
1. **Include in Git** (if under repo size limits)
2. **Use Railway Volumes** to upload separately  
3. **Cloud Storage** (S3/GCS) with download on startup
4. **Git LFS** for large files

## Alternative: Render

1. **Sign up at [render.com](https://render.com)**
2. **Create Web Service** from GitHub
3. **Environment**: Python 3
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `python deploy.py`
6. **Set same environment variables as above**

## Alternative: Vercel

1. **Install Vercel CLI**: `npm i -g vercel`
2. **Run**: `vercel` in your project directory
3. **Configure** for Python runtime
4. **Set environment variables** in Vercel dashboard

## Environment Variables Required

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
OPENAI_API_KEY=sk-your_api_key_here
MIXPANEL_TOKEN=your_token_here (optional for analytics)
FLASK_SECRET_KEY=random_secret_key_for_sessions
```

## Getting Spotify Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Set redirect URI to: `https://your-deployed-url.com/callback`
4. Copy Client ID and Client Secret

## Testing Deployment

After deployment:
1. Visit your app URL
2. Try a text search (requires OpenAI API key)
3. Login with Spotify
4. Try song-to-song search
5. Create a test playlist

## Troubleshooting

- **500 errors**: Check environment variables are set correctly
- **Data not loading**: Verify data files are uploaded/accessible
- **Spotify auth fails**: Check redirect URI matches deployed URL
- **Out of memory**: Use separate embedding files instead of combined .npz

## Cost Estimates (as of 2025)

- **Railway**: ~$5/month hobby plan
- **Render**: Free tier available, $7/month for paid
- **Vercel**: Free tier generous, $20/month pro
- **Fly.io**: ~$3-10/month depending on usage

Railway is recommended for the best balance of simplicity, features, and cost.