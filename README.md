# Semantic Song Search

A music similarity search engine built entirely from internet text data.

**Try the app here**: https://songmatch.up.railway.app

**Read the blog post**: https://andrewsingh.github.io/posts/text-not-tracks/

<img width="1000" alt="songmatch_screenshot" src="https://github.com/user-attachments/assets/a0c67fea-6798-4944-b259-3416becab8cc" />

## Overview

This is a web application that implements music similarity search purely from internet text data - no audio content or user history needed. It obtains semantic representations from only track metadata by first building structured text profiles of tracks and artists, then creating dense representations of these profiles to power embedding-based similarity search.

## Data Pipeline: Building Your Own Music Library

This section explains how to build your own text-based music similarity search from scratch using your own music library. The pipeline transforms track metadata into semantic representations through LLM-generated profiles and embeddings.

### Pipeline Overview

```
Spotify Metadata → Song Profiles → Embeddings → Search Engine
     (API)        (Perplexity)     (OpenAI)     (Flask App)
```

### Step 1: Fetch Track Metadata

**Script:** `scripts/fetch_spotify_metadata.py`

Fetches track metadata from Spotify Web API for your library of songs.

```bash
python scripts/fetch_spotify_metadata.py \
  --input your_track_ids.txt \
  --output data/your_library_metadata.json
```

**Output:** JSON array of track objects with fields:
- Track name, artist names, release date
- Spotify track ID and URI
- Total streams and daily streams (supplemented data)

### Step 2: Generate Song Profiles

**Script:** `scripts/generate_song_profiles.py`

Uses Perplexity Sonar Pro to search the web and generate structured text profiles for each track.

```bash
python scripts/generate_song_profiles.py \
  --input data/your_library_metadata.json \
  --output data/your_library_profiles.jsonl \
  --prompt prompts/your_song_profile_prompt.txt \
  --perplexity-api-key YOUR_PERPLEXITY_KEY
```

**Profile Structure:** Each song gets 6 sections:
1. **Genres** - Genre labels (e.g., "pop, synth-pop, dance-pop")
2. **Vocal Style** - Vocal characteristics (e.g., "breathy vocals, powerful belting")
3. **Production & Sound Design** - Sonic elements (e.g., "lush synths, tight drums")
4. **Lyrical Meaning** - Themes and content (e.g., "self-empowerment, confidence")
5. **Mood & Atmosphere** - Emotional qualities (e.g., "uplifting, energetic")
6. **Tags** - Single-word descriptors (e.g., "catchy, danceable, summery")

**Features:**
- Resume capability - skips already-processed songs
- Rate limiting and retry logic for API stability
- Cost: ~$18 per 1,000 tracks


### Step 3: Generate Artist Profiles

**Script:** `scripts/generate_artist_profiles.py`

Generates similar structured profiles for artists in your library.

```bash
python scripts/generate_artist_profiles.py \
  --input data/your_library_metadata.json \
  --output data/your_library_artist_profiles.jsonl \
  --prompt prompts/your_artist_profile_prompt.txt \
  --perplexity-api-key YOUR_PERPLEXITY_KEY
```

**Key Difference:** Artist genres include prominence scores (1-10) to weight how prominent each genre is in the artist's catalog.

**Cost:** ~$2.27 per 100 artists

### Step 4: Embed Song Profiles

**Script:** `scripts/embed_song_profiles.py`

Generates OpenAI embeddings for each section of each song profile.

```bash
python scripts/embed_song_profiles.py \
  --input data/your_library_profiles.jsonl \
  --output data/your_library_embeddings/ \
  --batch_size 500
```

**Output:** 6 separate `.npz` files, one per aspect:
- `genres_embeddings.npz`
- `vocal_style_embeddings.npz`
- `production_sound_design_embeddings.npz`
- `lyrical_meaning_embeddings.npz`
- `mood_atmosphere_embeddings.npz`
- `tags_embeddings.npz`

**Features:**
- Batch processing with resume capability
- Concurrent requests with rate limiting

### Step 5: Embed Artist Profiles

**Script:** `scripts/embed_artist_profiles.py`

Similar to song embedding, but handles artist profile sections and prominence-weighted genres.

```bash
python scripts/embed_artist_profiles.py \
  --input data/your_library_artist_profiles.jsonl \
  --output data/your_library_artist_embeddings/
```

### Step 6: Run the Search Engine

Point the Flask app to your generated data:

```bash
cd semantic_song_search
python app.py \
  --songs ../data/your_library_metadata.json \
  --embeddings ../data/your_library_embeddings \
  --debug
```

The search engine will automatically load your embeddings and metadata, and you can start searching!

### Pipeline Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `fetch_spotify_metadata.py` | Get track metadata | Track IDs | JSON metadata |
| `generate_song_profiles.py` | Generate song profiles | Metadata JSON | Profiles JSONL |
| `generate_artist_profiles.py` | Generate artist profiles | Metadata JSON | Artist profiles JSONL |
| `embed_song_profiles.py` | Create song embeddings | Profiles JSONL | 6× NPZ files |
| `embed_artist_profiles.py` | Create artist embeddings | Artist profiles JSONL | 5× NPZ files |

### Cost Estimates (2025 Pricing)

For a library of **6,000 songs** with **400 artists**:
- Song profile generation: ~$108 (Perplexity)
- Artist profile generation: ~$9 (Perplexity)
- Song embedding: ~$0.26 (OpenAI)
- Artist embedding: ~$0.03 (OpenAI)
- **Total: ~$117**

## Running the Web App

### Prerequisites
- Python 3.8+
- Spotify Developer Account (for client ID/secret)
- Perplexity API key (for generating song/artist profiles)
- OpenAI API key (for embedding profiles)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-song-search.git
cd semantic-song-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export SPOTIFY_CLIENT_ID="your_spotify_client_id"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
export OPENAI_API_KEY="your_openai_api_key"
export FLASK_SECRET_KEY="your_secret_key"  # Optional
export MIXPANEL_TOKEN="your_mixpanel_token"  # Optional for analytics
```

4. Run the application:
```bash
cd semantic_song_search
python app.py --debug --host 127.0.0.1 --port 5000
```

5. Open your browser to `http://127.0.0.1:5000`

### Using the App

1. **Login with Spotify** - Click the login button to authenticate
2. **Search for music** - Enter a text description or song name
3. **Filter results** - Use artist filters or Top Artists to refine
4. **Listen and explore** - Play songs directly in the browser
5. **Create playlist** - Export your favorite results to Spotify

## Features

### 🔍 Dual Search Modes

- **Text-to-Song**: Describe the vibe you're looking for
  - *"upbeat summer dance pop"*
  - *"laid-back chill EDM"*
  - *"motivational workout hip hop"*

- **Song-to-Song**: Find songs similar to one you already have in mind
  - Search by song and artist name
  - Discover music that matches the vibe of your favorite songs

### 🎯 Multi-Aspect Similarity Search

Search across 6 different musical dimensions:
1. Genres
2. Vocal Style
3. Production & Sound Design
4. Lyrical Meaning
5. Mood & Atmosphere
6. Tags

### 🎧 Interactive Features

- **Built-in Spotify Player** - Listen to songs directly in the app
- **Artist Filtering** - Multi-select dropdown to filter by specific artists
- **Top Artists Filter** - Show only songs from your top Spotify artists
- **Manual Selection** - Pick specific songs for your playlist

### 🚀 Spotify Integration

- **OAuth Authentication** - Secure login with Spotify
- **Playlist Export** - Create playlists directly in your Spotify library
- **Playback Control** - Full web player integration with queue management

## Technical Architecture

### Backend (Python/Flask)
- **Search Engine** (`search.py`) - Multi-aspect similarity matching with 6 embedding types
- **Ranking Engine** (`ranking.py`) - Future personalization system (currently pure semantic ranking)
- **REST API** - Comprehensive endpoints for search, filtering, and playlist creation

### Frontend (JavaScript)
- **Modular Architecture** - 7 specialized JavaScript modules
- **Performance Optimized** - DOM caching, Set/Map data structures for O(1) operations

## Development

### Running Tests
```bash
# Run regression tests with deterministic results
cd tests
PYTHONHASHSEED=42 python generate_results.py --input inputs/basic_queries.json --output-dir test_output --verbose
```

### Project Structure
```
semantic_song_search/
├── app.py                  # Main Flask application
├── search.py              # Search engine implementation
├── ranking.py             # Ranking and personalization
├── data_utils.py          # Data loading utilities
├── static/
│   ├── app.js            # Main application logic
│   ├── search.js         # Search functionality
│   ├── results-ui.js     # Results display
│   ├── spotify-player.js # Player integration
│   ├── playlist-export.js # Playlist creation
│   ├── personalization.js # Future personalization UI
│   └── utils.js          # Shared utilities
└── templates/
    └── index.html        # Main SPA template
```

## Deployment

### Deploying to Railway

Railway is the recommended platform for deploying your music similarity search app.

**Prerequisites:**
- GitHub account with your code pushed
- Spotify Developer App (Client ID and Secret)
- OpenAI API key
- Your generated data files from the pipeline

**Steps:**

1. **Sign up at [railway.app](https://railway.app)** and connect your GitHub

2. **Create a new project** from your GitHub repository

3. **Add a Volume** for your data files:
   - Go to your service → Settings → Volumes
   - Create a new volume mounted at `/app/data`
   - Upload your generated embedding files and metadata

4. **Set Environment Variables**:
   ```bash
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   OPENAI_API_KEY=your_openai_key
   FLASK_SECRET_KEY=random_secret_string
   MIXPANEL_TOKEN=your_token (optional)
   ```

5. **Update Spotify Redirect URI**:
   - Go to your Spotify Developer Dashboard
   - Add `https://your-app.railway.app/callback` to redirect URIs

6. **Deploy**: Railway will automatically build and deploy

**Cost:** ~$5/month on Railway's hobby plan

For detailed deployment instructions and alternatives (Render, Vercel), see `DEPLOYMENT.md`.


## Acknowledgments

- Built with Flask, Spotify Web API, Perplexity Sonar Pro, and OpenAI embeddings
- Uses Mixpanel for analytics tracking
- Hosted on Railway
