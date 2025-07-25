# 🎵 Intelligent Song Search

An AI-powered music discovery app that combines semantic search with Spotify integration for intelligent music exploration.

## Features

### 🔍 Dual Search Modes
- **Text-to-Song**: Search using natural language queries (e.g., "energetic dance pop", "melancholic indie")
- **Song-to-Song**: Find songs similar to ones you already know and love

### 🧠 AI-Powered Similarity
- **5 Embedding Types**: Choose how to match songs based on different aspects
  - Full Profile: Complete song information
  - Sound Aspect: Musical characteristics and production
  - Meaning Aspect: Lyrical themes and narrative
  - Mood Aspect: Emotional tone and atmosphere
  - Tags + Genres: Musical categorization

### 🎧 Spotify Integration
- **Web Player**: Play songs directly in the app
- **OAuth Authentication**: Secure Spotify login
- **Progress Control**: Seek, play/pause, skip tracks
- **Premium Required**: Spotify Premium needed for playback

### 🎨 Modern Interface
- **Dark Theme**: Easy on the eyes for long listening sessions
- **Responsive Design**: Works on desktop and mobile
- **Interactive Cards**: Hover effects and visual feedback
- **Real-time Suggestions**: Fuzzy matching for song selection

## Setup

### Prerequisites
- Python 3.8+
- Spotify Premium account
- OpenAI API key
- Spotify Developer account

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SPOTIFY_CLIENT_ID="your_spotify_client_id"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. Spotify App Configuration
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add `http://127.0.0.1:5000/callback` to Redirect URIs
   - If using a custom port, update accordingly (e.g., `http://127.0.0.1:8080/callback`)
   - **Important**: Use `127.0.0.1` not `localhost` for reliable OAuth redirects
   - **Critical**: The redirect URI must match exactly - including protocol, host, port, and path
4. Copy Client ID and Client Secret to environment variables

### 3. Data Files
**Default data files (in parent directory):**
- `../pop_eval_set_v0/pop_eval_set_v0_results_enriched.json` - Song profiles with metadata
- `../pop_eval_set_v0/pop_eval_set_v0_embeddings.npz` - Pre-computed embeddings

**Verify data files (optional but recommended):**
```bash
# Verify default data files
python verify_data.py

# Verify custom data files
python verify_data.py --songs /path/to/your/songs.json --embeddings /path/to/your/embeddings.npz
```

### 4. Run the App

**Basic usage (with default data files):**
```bash
python app.py
```

**With custom data files:**
```bash
python app.py --songs /path/to/your/songs.json --embeddings /path/to/your/embeddings.npz
```

**Full command line options:**
```bash
python app.py --help
```

**Common usage examples:**
```bash
# Default settings
python app.py

# Custom data files with short flags
python app.py -s custom_songs.json -e custom_embeddings.npz

# Run on different host/port
python app.py --host 0.0.0.0 --port 8080

# Enable debug mode
python app.py --debug

# Combine options
python app.py --songs /path/to/songs.json --embeddings /path/to/embeddings.npz --debug --port 8080
```

Navigate to `http://127.0.0.1:5000` in your browser (or your custom host:port).

## Command Line Arguments

The app supports several command line arguments for flexibility:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--songs` | `-s` | `../pop_eval_set_v0/pop_eval_set_v0_results_enriched.json` | Path to song profiles JSON file |
| `--embeddings` | `-e` | `../pop_eval_set_v0/pop_eval_set_v0_embeddings.npz` | Path to embeddings NPZ file |
| `--host` | | `127.0.0.1` | Host to run the server on |
| `--port` | | `5000` | Port to run the server on |
| `--debug` | | `False` | Enable debug mode |
| `--help` | `-h` | | Show help message |

**Examples:**
```bash
# See all options
python app.py --help

# Use custom dataset
python app.py -s my_songs.json -e my_embeddings.npz

# Run on all interfaces with debug mode (requires updating Spotify redirect URI)
python app.py --host 0.0.0.0 --debug

# Production-like setup (requires updating Spotify redirect URI)
python app.py --host 0.0.0.0 --port 80

# Custom port with default local access
python app.py --port 8080
```

**Important for Spotify OAuth:**
- If you change the host or port, you **must** update the redirect URI in your [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- For `--host 0.0.0.0 --port 8080`, add `http://0.0.0.0:8080/callback` to Redirect URIs
- The redirect URI must match exactly what the app is running on

## Usage

### Text-to-Song Search
1. Select "Text-to-Song" from the search type dropdown
2. Choose an embedding type (e.g., "Mood Aspect" for emotional similarity)
3. Enter a descriptive query like "upbeat summer vibes" or "melancholic piano ballad"
4. Press Enter or click Search
5. Browse results and click any song to play

### Song-to-Song Search
1. Select "Song-to-Song" from the search type dropdown
2. Choose an embedding type (e.g., "Sound Aspect" for musical similarity)
3. Start typing a song or artist name
4. Select from the dropdown suggestions
5. View similar songs and click to play

### Spotify Player
- **Connect**: Login with your Spotify Premium account
- **Play**: Click any song card to start playback
- **Controls**: Use play/pause, previous/next buttons
- **Seek**: Click anywhere on the progress bar to jump to that position

## Technical Details

### Architecture
- **Backend**: Flask web server with RESTful API
- **Frontend**: Vanilla JavaScript with modern ES6+ features
- **Search Engine**: Custom similarity search using cosine similarity
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Song Matching**: TF-IDF vectorization for fuzzy text matching

### API Endpoints
- `GET /` - Main application page
- `GET /login` - Spotify OAuth login
- `GET /callback` - OAuth callback handler
- `GET /api/search_suggestions` - Get song suggestions for song-to-song search
- `POST /api/search` - Main search endpoint
- `GET /api/get_song` - Get detailed song information
- `GET /api/token` - Get current Spotify access token

### Data Structure
The app uses pre-computed embeddings organized by type:
- **Songs**: 1,471 tracks with rich metadata
- **Embeddings**: 7,355 total embeddings (5 per song on average)
- **Types**: full_profile, sound_aspect, meaning_aspect, mood_aspect, tags_genres

## Customization

### Adding New Embedding Types
1. Generate embeddings using the pattern in `embed_song_profiles.py`
2. Update the `EMBEDDING_TYPES` in the search engine
3. Add new options to the frontend dropdown

### Styling
- Modify `static/style.css` for visual customization
- Colors, fonts, layouts are all configurable
- Responsive breakpoints can be adjusted

### Search Parameters
- Adjust `k` parameter in search requests for more/fewer results
- Modify similarity thresholds in the search engine
- Customize TF-IDF parameters for better text matching

## Troubleshooting

### Common Issues
1. **"Not connected to Spotify"**: Ensure Spotify credentials are correct and redirect URI is configured
2. **"Player not ready"**: Check that you have Spotify Premium and the Web Playback SDK loaded
3. **"Search failed"**: Verify OpenAI API key and check network connectivity
4. **No suggestions**: Ensure the song profile data is loaded correctly

### Logs
Check the Flask console for detailed error messages and debugging information.

## Future Enhancements

- [ ] Playlist creation from search results
- [ ] User preferences and search history
- [ ] Advanced filtering (year, genre, energy level)
- [ ] Social features (share searches, collaborative playlists)
- [ ] Mobile app version
- [ ] More music sources beyond Spotify

## License

This project is for educational and personal use. Ensure compliance with Spotify and OpenAI terms of service. 