// Spotify Player Module for Semantic Song Search

class SpotifyPlayer {
    constructor(app) {
        this.app = app;
        
        // Player state
        this.spotifyPlayer = null;
        this.deviceId = null;
        this.isPlayerReady = false;
        this.playerInitPromise = null;
        this.currentTrack = null;
        this.lastTrackId = null;
        
        // Auto-play queue management
        this.currentSongIndex = -1;
        this.isAutoPlayEnabled = true;
        this.isManualSkip = false;
        this.autoAdvancePending = false;
        this.lastAutoAdvanceTime = 0;
        this.isPlayingSong = false;
        this.autoPlayCheckInterval = null;
        this.lastProcessedTrackEnd = null;
    }

    get accessToken() {
        return this.app.accessToken;
    }

    get searchResults() {
        return this.app.searchResults;
    }

    get searchResultsId() {
        return this.app.searchResultsId;
    }

    async initSpotifyPlayer() {
        console.log('🎧 initSpotifyPlayer called');
        console.log('🎧 Access token available?', this.accessToken ? 'Yes' : 'No');
        
        if (!this.accessToken) {
            return Promise.reject(new Error('No access token'));
        }
        
        if (!window.Spotify) {
            console.log('⏳ Spotify SDK not loaded, retrying in 1 second...');
            return new Promise((resolve, reject) => {
                setTimeout(() => {
                    this.initSpotifyPlayer().then(resolve).catch(reject);
                }, 1000);
            });
        }
        
        // Return existing promise if initialization is already in progress
        if (this.playerInitPromise) {
            console.log('🎧 Player initialization already in progress, returning existing promise');
            return this.playerInitPromise;
        }
        
        // Create a promise for player initialization
        this.playerInitPromise = new Promise((resolve, reject) => {
            const player = new window.Spotify.Player({
                name: 'Semantic Song Search',
                getOAuthToken: cb => { 
                    console.log('🔑 Token requested by Spotify SDK');
                    cb(this.accessToken); 
                },
                volume: 0.5
            });
            
            // Error handling
            player.addListener('initialization_error', ({ message }) => {
                console.error('❌ Failed to initialize:', message);
                this.playerInitPromise = null;
                reject(new Error(`Initialization error: ${message}`));
            });
            
            player.addListener('authentication_error', ({ message }) => {
                console.error('❌ Failed to authenticate:', message);
                this.app.updateAuthStatus(false);
                this.playerInitPromise = null;
                reject(new Error(`Authentication error: ${message}`));
            });
            
            player.addListener('account_error', ({ message }) => {
                console.error('❌ Failed to validate Spotify account:', message);
                this.playerInitPromise = null;
                reject(new Error(`Account error: ${message}`));
            });
            
            player.addListener('playback_error', ({ message }) => {
                console.error('❌ Failed to perform playback:', message);
            });
            
            // Playback status updates
            player.addListener('player_state_changed', (state) => {
                if (!state) return;
                
                if (state.track_window.current_track) {
                    console.log('🎵 State details:', {
                        paused: state.paused,
                        position: state.position,
                        duration: state.duration,
                        loading: state.loading,
                        trackId: state.track_window.current_track.id,
                        autoPlayEnabled: this.isAutoPlayEnabled,
                        currentIndex: this.currentSongIndex,
                        manualSkip: this.isManualSkip,
                        timeSinceLastAdvance: Date.now() - this.lastAutoAdvanceTime
                    });
                }
                
                this.updatePlayerUI(state);
                this.updatePlayingCards(state.track_window.current_track);
                
                // Auto-play next song when current track ends
                this.handleAutoPlayCheck(state);
            });
            
            // Ready
            player.addListener('ready', ({ device_id }) => {
                this.deviceId = device_id;
                this.spotifyPlayer = player;
                this.isPlayerReady = true;
                this.playerInitPromise = null;
                
                resolve(device_id);
            });
            
            // Not Ready
            player.addListener('not_ready', ({ device_id }) => {
                this.isPlayerReady = false;
                console.log('🎧 Device ID has gone offline:', device_id);
            });
            
            // Connect to the player
            player.connect().then(success => {
                if (success) {
                    console.log('🎧 Successfully connected to Spotify!');
                } else {
                    console.log('🎧 Failed to connect to Spotify');
                    this.playerInitPromise = null;
                    reject(new Error('Failed to connect to Spotify'));
                }
            });
        });
        
        return this.playerInitPromise;
    }

    async playSong(song, isAutoAdvance = false) {
        console.log(`🎵 Playing song: "${song.song}" by ${song.artist}${isAutoAdvance ? ' (auto-advance)' : ''}`);
        
        if (!song.spotify_id) {
            console.error('❌ Song has no Spotify ID:', song);
            return;
        }
        
        if (!this.accessToken) {
            console.error('❌ No access token available');
            return;
        }
        
        // Prevent concurrent playSong calls
        if (this.isPlayingSong) {
            console.log('🎵 Already playing a song, skipping...');
            return;
        }
        
        this.isPlayingSong = true;
        
        try {
            // Initialize player if not ready
            if (!this.isPlayerReady) {
                console.log('🎧 Player not ready, initializing...');
                await this.initSpotifyPlayer();
            }
            
            // Play the track
            const response = await fetch(`https://api.spotify.com/v1/me/player/play?device_id=${this.deviceId}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    uris: [`spotify:track:${song.spotify_id}`]
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ Spotify API error:', response.status, errorText);
                throw new Error(`Spotify API error: ${response.status} - ${errorText}`);
            }
            
            this.currentTrack = song;
            this.lastTrackId = song.spotify_id;
            
            // Track song play
            this.app.analytics.trackEvent('Song Played', {
                'song_spotify_id': song.spotify_id || 'unknown',
                'song_title': song.song || 'unknown',
                'artist': song.artist || 'unknown',
                'play_method': isAutoAdvance ? 'auto_advance' : 'manual_click',
                'similarity_score': song.similarity || 0,
                'position_in_results': this.searchResults.findIndex(r => r.song_idx === song.song_idx) + 1,
                'is_authenticated': this.app.isAuthenticated
            });
            
            if (!isAutoAdvance) {
                this.app.analytics.incrementSongsPlayed();
            }
            
            // Update current song index for queue management
            if (isAutoAdvance) {
                // For auto-advance, we've already set the index in nextTrack/previousTrack
                this.lastAutoAdvanceTime = Date.now();
            } else {
                // For manual play, find the index in current results
                const songIndex = this.searchResults.findIndex(r => r.song_idx === song.song_idx);
                if (songIndex !== -1) {
                    this.currentSongIndex = songIndex;
                    console.log(`🎵 Updated current song index to ${this.currentSongIndex}`);
                }
            }
            
            // Reset manual skip flag and other state
            this.isManualSkip = false;
            this.autoAdvancePending = false;
            
            // Reset the playing flag after successful start
            setTimeout(() => {
                this.isPlayingSong = false;
                this.lastProcessedTrackEnd = null;
                if (this.isManualSkip) {
                    this.isManualSkip = false;
                }
            }, 1000);
            
        } catch (error) {
            console.error('❌ Error playing song:', error);
            
            // Track song play failures
            this.app.analytics.trackEvent('Song Play Failed', {
                'song_spotify_id': song.spotify_id || 'unknown',
                'song_title': song.song || 'unknown',
                'artist': song.artist || 'unknown',
                'error_message': error.message || 'Unknown error',
                'play_method': isAutoAdvance ? 'auto_advance' : 'manual_click'
            });
            
            // Reset current song index on error
            this.currentSongIndex = -1;
            
            this.app.showError('Failed to play song. Please check your Spotify connection.');
            this.isPlayingSong = false;
        }
    }

    async ensureDeviceActive() {
        try {
            const response = await fetch('https://api.spotify.com/v1/me/player/devices', {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`Spotify API error: ${response.status}`);
            }
            
            const data = await response.json();
            if (!data.devices) {
                throw new Error('No devices found in Spotify API response');
            }
            
            const ourDevice = data.devices.find(device => device.id === this.deviceId);
            
            if (!ourDevice || !ourDevice.is_active) {
                // Transfer playback to our device
                await fetch('https://api.spotify.com/v1/me/player', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.accessToken}`
                    },
                    body: JSON.stringify({
                        device_ids: [this.deviceId],
                        play: false
                    })
                });
            }
        } catch (error) {
            console.error('Error ensuring device active:', error);
        }
    }

    async togglePlayback() {
        if (!this.spotifyPlayer) {
            console.log('🎧 No player available');
            return;
        }
        
        try {
            await this.spotifyPlayer.togglePlay();
        } catch (error) {
            console.error('❌ Error toggling playback:', error);
        }
    }

    async previousTrack() {
        const isAutoAdvance = false;
        this.isManualSkip = true;
        
        if (this.currentSongIndex <= 0 || this.searchResults.length === 0) {
            console.log('🎵 Already at first song or no results');
            return;
        }
        
        this.currentSongIndex--;
        const prevSong = this.searchResults[this.currentSongIndex];
        if (prevSong) {
            await this.playSong(prevSong, isAutoAdvance);
        }
    }

    async nextTrack() {
        const isAutoAdvance = false;
        this.isManualSkip = true;
        
        if (this.currentSongIndex >= this.searchResults.length - 1 || this.searchResults.length === 0) {
            console.log('🎵 Already at last song or no results');
            return;
        }
        
        this.currentSongIndex++;
        const nextSong = this.searchResults[this.currentSongIndex];
        if (nextSong) {
            await this.playSong(nextSong, isAutoAdvance);
        }
    }

    handleAutoPlayCheck(state, isBackupCheck = false) {
        // Basic checks - exit early if conditions not met
        if (!state || !this.isAutoPlayEnabled || this.currentSongIndex < 0 || this.isPlayingSong || this.autoAdvancePending) {
            return;
        }
        
        const currentTrack = state.track_window.current_track;
        if (!currentTrack) return;
        
        if (!isBackupCheck) {
            console.log('🎵 Track info:', {
                currentTrackId: currentTrack.id,
                lastTrackId: this.lastTrackId,
                trackChanged: currentTrack.id !== this.lastTrackId,
                manualSkipFlag: this.isManualSkip
            });
        }
        
        // Rate limiting - don't check too frequently
        const now = Date.now();
        if (now - this.lastAutoAdvanceTime < 1000) {
            return;
        }
        
        // Multiple ways to detect track ending
        const position = state.position || 0;
        const duration = state.duration || 0;
        const timeRemaining = duration - position;
        const trackEnded = timeRemaining <= 1000; // Within 1 second of ending
        const trackStopped = state.paused && position === 0 && !state.loading;
        const trackChanged = currentTrack.id !== this.lastTrackId && this.lastTrackId;
        
        if (!isBackupCheck) {
            console.log('🎵 Auto-play conditions:', {
                trackEnded, trackStopped, trackChanged,
                timeRemaining, position, duration,
                paused: state.paused, loading: state.loading,
                hasMoreSongs: this.currentSongIndex < this.searchResults.length - 1,
                isManualSkip: this.isManualSkip
            });
        }
        
        // Detect if track has ended or changed
        const shouldAdvance = (trackEnded || trackStopped || trackChanged) && !this.isManualSkip;
        
        if (shouldAdvance && this.currentSongIndex < this.searchResults.length - 1) {
            // Prevent duplicate processing of the same track end
            const trackEndId = `${currentTrack.id}_${Math.floor(position / 1000)}`;
            if (this.lastProcessedTrackEnd === trackEndId) {
                return;
            }
            this.lastProcessedTrackEnd = trackEndId;
            
            this.autoAdvancePending = true;
            
            console.log('🎵 Auto-advancing to next track...');
            
            this.currentSongIndex++;
            const nextSong = this.searchResults[this.currentSongIndex];
            if (nextSong) {
                const isAutoAdvance = true;
                this.playSong(nextSong, isAutoAdvance);
            }
        } else if (this.isManualSkip) {
            // Reset manual skip flag for the next track
            setTimeout(() => {
                console.log('🎵 Resetting manual skip flag after track change');
                this.isManualSkip = false;
            }, 2000);
        }
    }

    updatePlayerUI(state) {
        if (!state) return;
        
        // Update play/pause button
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {
            playBtn.textContent = state.paused ? '▶' : '⏸';
        }
        
        // Update track metadata
        if (state.track_window && state.track_window.current_track) {
            const track = state.track_window.current_track;
            
            // Update track title
            const playerTitle = document.getElementById('player-title');
            if (playerTitle) {
                playerTitle.textContent = track.name || 'Unknown Track';
            }
            
            // Update artist name
            const playerArtist = document.getElementById('player-artist');
            if (playerArtist) {
                const artists = track.artists.map(artist => artist.name).join(', ');
                playerArtist.textContent = artists || 'Unknown Artist';
            }
            
            // Update cover image
            const playerCover = document.getElementById('player-cover');
            if (playerCover && track.album && track.album.images && track.album.images.length > 0) {
                playerCover.src = track.album.images[0].url;
                playerCover.alt = `${track.album.name} cover`;
            }
        }
        
        // Update progress
        if (state.position !== undefined && state.duration !== undefined) {
            this.updateProgress(state.position, state.duration);
        }
    }

    updateProgress(position, duration) {
        const progressBar = document.getElementById('progress-bar');
        const progressFill = document.getElementById('progress-filled');
        const currentTimeEl = document.getElementById('current-time');
        const totalTimeEl = document.getElementById('total-time');
        
        if (progressFill && duration > 0) {
            const percentage = (position / duration) * 100;
            progressFill.style.width = `${percentage}%`;
        }
        
        if (currentTimeEl) {
            currentTimeEl.textContent = formatTime(position);
        }
        
        if (totalTimeEl) {
            totalTimeEl.textContent = formatTime(duration);
        }
    }

    updatePlayingCards(currentTrack) {
        if (!currentTrack) return;
        
        // Remove playing state from all cards
        const allCards = document.querySelectorAll('.song-card');
        allCards.forEach(card => {
            card.classList.remove('playing');
            const playIcon = card.querySelector('.play-icon');
            if (playIcon) {
                playIcon.textContent = '▶';
            }
        });
        
        // Add playing state to current card
        const currentCard = document.querySelector(`[data-spotify-id="${currentTrack.id}"]`);
        if (currentCard) {
            currentCard.classList.add('playing');
            const playIcon = currentCard.querySelector('.play-icon');
            if (playIcon) {
                playIcon.textContent = '🎵';
            }
        }
    }

    resetAutoPlayQueue() {
        console.log('🎵 Resetting auto-play queue');
        this.currentSongIndex = -1;
        this.isManualSkip = false;
        this.autoAdvancePending = false;
        this.lastTrackId = null;
        this.lastProcessedTrackEnd = null;
        
        // Remove playing state from all cards
        const allCards = document.querySelectorAll('.song-card');
        allCards.forEach(card => {
            card.classList.remove('playing');
            const playIcon = card.querySelector('.play-icon');
            if (playIcon) {
                playIcon.textContent = '▶';
            }
        });
    }

    async seekToPosition(event) {
        if (!this.spotifyPlayer || !this.isPlayerReady) {
            console.log('🎧 Player not ready for seeking');
            return;
        }
        
        try {
            const progressBar = event.currentTarget;
            const rect = progressBar.getBoundingClientRect();
            const percentage = (event.clientX - rect.left) / rect.width;
            
            // Get current state to determine duration
            const state = await this.spotifyPlayer.getCurrentState();
            if (state && state.duration) {
                const newPosition = Math.floor(percentage * state.duration);
                await this.spotifyPlayer.seek(newPosition);
            }
        } catch (error) {
            console.error('❌ Error seeking:', error);
        }
    }
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpotifyPlayer;
}