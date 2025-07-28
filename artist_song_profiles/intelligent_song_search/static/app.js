// Intelligent Song Search App JavaScript

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class IntelligentSearchApp {
    constructor() {
        this.currentQuery = null;
        this.currentQuerySong = null;
        this.searchResults = [];
        this.currentSearchData = null;
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
        this.isLoadingMore = false;
        this.spotifyPlayer = null;
        this.deviceId = null;
        this.accessToken = null;
        this.currentTrack = null;
        this.isPlayerReady = false;
        
        this.init();
    }
    
    init() {
        this.bindEventListeners();
        this.checkAuthStatus();
        this.initSpotifyPlayer();
    }
    
    bindEventListeners() {
        // Search type change
        document.getElementById('search-type').addEventListener('change', (e) => {
            this.handleSearchTypeChange(e.target.value);
        });
        
        // Search input
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });
        
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSearch();
            }
        });
        
        // Search button
        document.getElementById('search-btn').addEventListener('click', () => {
            this.handleSearch();
        });
        
        // Login button
        document.getElementById('login-btn').addEventListener('click', () => {
            window.location.href = '/login';
        });
        
        // Player controls
        document.getElementById('play-btn').addEventListener('click', () => {
            this.togglePlayback();
        });
        
        document.getElementById('prev-btn').addEventListener('click', () => {
            this.previousTrack();
        });
        
        document.getElementById('next-btn').addEventListener('click', () => {
            this.nextTrack();
        });
        
        // Progress bar
        document.getElementById('progress-bar').addEventListener('click', (e) => {
            this.seekToPosition(e);
        });
        
        // Load more button
        document.getElementById('load-more-btn').addEventListener('click', () => {
            this.loadMoreResults();
        });
    }
    
    handleSearchTypeChange(searchType) {
        const suggestionsContainer = document.getElementById('suggestions');
        const querySection = document.getElementById('query-section');
        const searchInput = document.getElementById('search-input');
        
        if (searchType === 'song') {
            searchInput.placeholder = "🔍 Search for a song or artist... (e.g., 'Billie Eilish', 'Bad Guy')";
            this.clearResults();
        } else {
            searchInput.placeholder = "🔍 Search for music... (e.g., 'energetic dance pop', 'melancholic ballad')";
            suggestionsContainer.style.display = 'none';
            querySection.style.display = 'none';
            this.clearResults();
        }
    }
    
    async handleSearchInput(query) {
        const searchType = document.getElementById('search-type').value;
        const suggestionsContainer = document.getElementById('suggestions');
        
        if (searchType === 'song' && query.trim().length > 2) {
            try {
                const response = await fetch(`/api/search_suggestions?query=${encodeURIComponent(query)}`);
                const suggestions = await response.json();
                this.displaySuggestions(suggestions);
                suggestionsContainer.style.display = 'block';
            } catch (error) {
                console.error('Error fetching suggestions:', error);
                suggestionsContainer.style.display = 'none';
            }
        } else {
            suggestionsContainer.style.display = 'none';
        }
    }
    
    displaySuggestions(suggestions) {
        const container = document.getElementById('suggestions');
        container.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.onclick = () => this.selectSuggestion(suggestion);
            
            item.innerHTML = `
                <img src="${escapeHtml(suggestion.cover_url || '')}" alt="Cover" class="suggestion-cover">
                <div class="suggestion-details">
                    <div class="suggestion-title">${escapeHtml(suggestion.song)}</div>
                    <div class="suggestion-artist">${escapeHtml(suggestion.artist)}</div>
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    selectSuggestion(suggestion) {
        this.currentQuerySong = suggestion;
        document.getElementById('search-input').value = suggestion.label;
        document.getElementById('suggestions').style.display = 'none';
        this.displayQueryCard(suggestion);
        this.handleSearch();
    }
    
    displayQueryCard(song) {
        const querySection = document.getElementById('query-section');
        const queryCard = document.getElementById('query-card');
        
        queryCard.dataset.spotifyId = song.spotify_id; // Set the spotify ID for consistency
        queryCard.innerHTML = this.createSongCardHTML(song, { isQuery: true });
        querySection.style.display = 'block';
    }
    
    async handleSearch() {
        const searchType = document.getElementById('search-type').value;
        const embedType = document.getElementById('embed-type').value;
        const query = document.getElementById('search-input').value.trim();
        
        if (!query) return;
        
        // Reset pagination for new searches
        this.currentOffset = 0;
        this.searchResults = [];
        
        this.showLoading(true);
        this.hideWelcomeMessage();
        
        try {
            const requestData = {
                search_type: searchType,
                embed_type: embedType,
                query: query,
                k: 20,
                offset: 0
            };
            
            if (searchType === 'song' && this.currentQuerySong) {
                requestData.song_idx = this.currentQuerySong.song_idx;
            }
            
            // Store current search data for load more functionality
            this.currentSearchData = { ...requestData };
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.searchResults = data.results;
            this.currentOffset = data.pagination.offset + data.pagination.limit;
            this.totalResultsCount = data.pagination.total_count;
            this.hasMoreResults = data.pagination.has_more;
            this.displayResults(data, false);
            
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Search failed. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadMoreResults() {
        if (this.isLoadingMore || !this.hasMoreResults || !this.currentSearchData) {
            return;
        }
        
        this.isLoadingMore = true;
        this.showLoadMoreLoading(true);
        
        try {
            const requestData = {
                ...this.currentSearchData,
                offset: this.currentOffset
            };
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Load more failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.searchResults = [...this.searchResults, ...data.results];
            this.currentOffset = data.pagination.offset + data.pagination.limit;
            this.hasMoreResults = data.pagination.has_more;
            this.displayResults(data, true);
            
        } catch (error) {
            console.error('Load more error:', error);
            this.showError('Failed to load more results. Please try again.');
        } finally {
            this.isLoadingMore = false;
            this.showLoadMoreLoading(false);
        }
    }
    
    displayResults(data, isLoadMore = false) {
        const resultsHeader = document.getElementById('results-header');
        const resultsCount = document.getElementById('results-count');
        const searchInfo = document.getElementById('search-info');
        const resultsGrid = document.getElementById('results-grid');
        const loadMoreContainer = document.getElementById('load-more-container');
        
        // Update header (show total results, not just current batch)
        resultsCount.textContent = `${this.searchResults.length} of ${this.totalResultsCount} results`;
        searchInfo.textContent = `${data.search_type} search • ${data.embed_type.replace('_', ' ')}`;
        resultsHeader.style.display = 'flex';
        
        // Clear results grid only for new searches, not for load more
        if (!isLoadMore) {
            resultsGrid.innerHTML = '';
        }
        
        // Handle empty results case
        if (!isLoadMore && this.totalResultsCount === 0) {
            resultsGrid.innerHTML = `
                <div class="no-results-message">
                    <h3>No results found</h3>
                    <p>Try adjusting your search terms or selecting a different embedding type.</p>
                </div>
            `;
            loadMoreContainer.style.display = 'none';
            return;
        }
        
        // Add new results to the grid  
        const startIndex = isLoadMore ? this.searchResults.length - data.results.length : 0;
        data.results.forEach((song, index) => {
            const card = document.createElement('div');
            card.className = 'song-card';
            card.dataset.spotifyId = song.spotify_id; // Set the spotify ID for the updatePlayingCards function
            card.innerHTML = this.createSongCardHTML(song, { 
                rank: startIndex + index + 1, 
                similarity: song.similarity,
                fieldValue: song.field_value,
                embedType: data.embed_type
            });
            
            // Add click listener for playing song (but not on accordion elements)
            card.addEventListener('click', (e) => {
                // Don't play song if clicking on accordion elements
                if (!e.target.closest('.card-accordion')) {
                    this.playSong(song);
                }
            });
            
            // Add accordion toggle functionality
            const accordionToggle = card.querySelector('.accordion-toggle');
            if (accordionToggle) {
                accordionToggle.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent song play when clicking accordion
                    this.toggleAccordion(accordionToggle);
                });
            }
            
            resultsGrid.appendChild(card);
        });
        
        // Update load more button visibility
        this.updateLoadMoreButton();
    }
    
    updateLoadMoreButton() {
        const loadMoreContainer = document.getElementById('load-more-container');
        const loadMoreBtn = document.getElementById('load-more-btn');
        
        if (this.hasMoreResults && this.searchResults.length > 0) {
            loadMoreContainer.style.display = 'block';
            loadMoreBtn.disabled = this.isLoadingMore;
            loadMoreBtn.textContent = this.isLoadingMore ? 'Loading...' : 'Load More Results';
        } else {
            loadMoreContainer.style.display = 'none';
        }
    }
    
    showLoadMoreLoading(show) {
        this.updateLoadMoreButton();
    }
    
    createSongCardHTML(song, options = {}) {
        const { rank, similarity, isQuery = false, fieldValue = null, embedType = null } = options;
        
        let metadataHTML = '';
        if (rank && similarity !== undefined) {
            metadataHTML = `
                <div class="card-metadata">
                    <span>#${rank}</span>
                    <span class="similarity-score">${(similarity * 100).toFixed(1)}%</span>
                </div>
            `;
        }
        
        let tagsHTML = '';
        if (song.tags && song.tags.length > 0) {
            const displayTags = song.tags.slice(0, 3);
            tagsHTML = `
                <div class="card-tags">
                    ${displayTags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                </div>
            `;
        }
        
        // Accordion content based on embedding type
        let accordionHTML = '';
        if (fieldValue !== null && fieldValue !== undefined && embedType && !isQuery) {
            const accordionContent = this.formatFieldValueForDisplay(fieldValue, embedType);
            const accordionTitle = this.getAccordionTitle(embedType);
            
            accordionHTML = `
                <div class="card-accordion">
                    <button class="accordion-toggle" aria-expanded="false">
                        <span class="accordion-title">${accordionTitle}</span>
                        <span class="accordion-icon">▼</span>
                    </button>
                    <div class="accordion-content">
                        <div class="accordion-content-inner">
                            ${accordionContent}
                        </div>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="card-header">
                <img src="${escapeHtml(song.cover_url || '')}" alt="Cover" class="card-cover">
                <div class="card-info">
                    <div class="card-title">${escapeHtml(song.song)}</div>
                    <div class="card-artist">${escapeHtml(song.artist)}</div>
                    <div class="card-album">${escapeHtml(song.album || 'Unknown Album')}</div>
                </div>
            </div>
            ${metadataHTML}
            ${tagsHTML}
            ${accordionHTML}
        `;
    }
    
    formatFieldValueForDisplay(fieldValue, embedType) {
        /**
         * Format field value content for accordion display based on embedding type
         */
        // Handle null/undefined values
        if (fieldValue === null || fieldValue === undefined) {
            return '<div class="generic-content"><em>No content available</em></div>';
        }
        
        const escapedValue = escapeHtml(String(fieldValue)); // Ensure it's a string
        
        switch (embedType) {
            case 'full_profile':
                // Format the full profile with proper line breaks and structure
                return `<div class="profile-content">${escapedValue.replace(/\n/g, '<br>')}</div>`;
                
            case 'sound_aspect':
                return `<div class="aspect-content sound-aspect">${escapedValue}</div>`;
                
            case 'meaning_aspect':
                return `<div class="aspect-content meaning-aspect">${escapedValue}</div>`;
                
            case 'mood_aspect':
                return `<div class="aspect-content mood-aspect">${escapedValue}</div>`;
                
            case 'tags_genres':
                // Display tags and genres as a nicely formatted list
                // Handle empty or invalid strings gracefully
                if (!fieldValue || typeof fieldValue !== 'string') {
                    return '<div class="tags-genres-content"><em>No tags or genres available</em></div>';
                }
                
                const items = String(fieldValue).split(',').map(item => item.trim()).filter(item => item);
                
                if (items.length === 0) {
                    return '<div class="tags-genres-content"><em>No tags or genres available</em></div>';
                }
                
                return `
                    <div class="tags-genres-content">
                        ${items.map(item => `<span class="tag-genre-item">${escapeHtml(item)}</span>`).join('')}
                    </div>
                `;
                
            default:
                return `<div class="generic-content">${escapedValue}</div>`;
        }
    }
    
    getAccordionTitle(embedType) {
        /**
         * Get display title for accordion based on embedding type
         */
        switch (embedType) {
            case 'full_profile':
                return 'Full Profile';
            case 'sound_aspect':
                return 'Sound Description';
            case 'meaning_aspect':
                return 'Meaning & Lyrics';
            case 'mood_aspect':
                return 'Mood & Feeling';
            case 'tags_genres':
                return 'All Tags & Genres';
            default:
                return 'Details';
        }
    }
    
    toggleAccordion(toggleButton) {
        /**
         * Toggle accordion open/closed state
         */
        if (!toggleButton) {
            console.error('toggleAccordion: toggleButton is null');
            return;
        }
        
        const isExpanded = toggleButton.getAttribute('aria-expanded') === 'true';
        const accordionContent = toggleButton.nextElementSibling;
        const accordionIcon = toggleButton.querySelector('.accordion-icon');
        
        if (!accordionContent) {
            console.error('toggleAccordion: accordionContent not found');
            return;
        }
        
        if (!accordionIcon) {
            console.error('toggleAccordion: accordionIcon not found');
            return;
        }
        
        if (isExpanded) {
            // Collapse
            toggleButton.setAttribute('aria-expanded', 'false');
            accordionContent.style.maxHeight = '0px';
            accordionIcon.textContent = '▼';
            toggleButton.classList.remove('expanded');
        } else {
            // Expand
            toggleButton.setAttribute('aria-expanded', 'true');
            accordionContent.style.maxHeight = accordionContent.scrollHeight + 'px';
            accordionIcon.textContent = '▲';
            toggleButton.classList.add('expanded');
        }
    }
    
    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }
    
    hideWelcomeMessage() {
        document.getElementById('welcome-message').style.display = 'none';
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a proper error UI
        console.error(message);
        alert(message);
    }
    
    clearResults() {
        document.getElementById('results-header').style.display = 'none';
        document.getElementById('results-grid').innerHTML = '';
        document.getElementById('load-more-container').style.display = 'none';
        document.getElementById('welcome-message').style.display = 'block';
        this.currentQuerySong = null;
        this.searchResults = [];
        this.currentSearchData = null;
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
    }
    
    // Spotify Authentication and Player
    async checkAuthStatus() {
        try {
            const response = await fetch('/api/token');
            if (response.ok) {
                const data = await response.json();
                this.accessToken = data.access_token;
                this.updateAuthStatus(true);
            } else {
                this.updateAuthStatus(false);
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            this.updateAuthStatus(false);
        }
    }
    
    updateAuthStatus(isAuthenticated) {
        const indicator = document.getElementById('auth-indicator');
        const text = document.getElementById('auth-text');
        const loginBtn = document.getElementById('login-btn');
        
        if (isAuthenticated) {
            indicator.textContent = '●';
            indicator.className = 'auth-indicator connected';
            text.textContent = 'Connected to Spotify';
            loginBtn.style.display = 'none';
        } else {
            indicator.textContent = '○';
            indicator.className = 'auth-indicator';
            text.textContent = 'Not connected to Spotify';
            loginBtn.style.display = 'inline-block';
        }
    }
    
    initSpotifyPlayer() {
        console.log('🎧 initSpotifyPlayer called');
        console.log('🎧 Access token available?', this.accessToken ? 'Yes' : 'No');
        
        if (!this.accessToken) {
            console.log('❌ No access token, skipping player initialization');
            return;
        }
        
        if (!window.Spotify) {
            console.log('⏳ Spotify SDK not loaded, retrying in 1 second...');
            setTimeout(() => this.initSpotifyPlayer(), 1000);
            return;
        }
        
        console.log('✅ Spotify SDK loaded, creating player...');
        
        const player = new window.Spotify.Player({
            name: 'Intelligent Song Search',
            getOAuthToken: cb => { 
                console.log('🔑 Token requested by Spotify SDK');
                cb(this.accessToken); 
            },
            volume: 0.5
        });
        
        // Error handling
        player.addListener('initialization_error', ({ message }) => {
            console.error('❌ Failed to initialize:', message);
        });
        
        player.addListener('authentication_error', ({ message }) => {
            console.error('❌ Failed to authenticate:', message);
            this.updateAuthStatus(false);
        });
        
        player.addListener('account_error', ({ message }) => {
            console.error('❌ Failed to validate Spotify account:', message);
        });
        
        player.addListener('playback_error', ({ message }) => {
            console.error('❌ Failed to perform playback:', message);
        });
        
        // Playback status updates
        player.addListener('player_state_changed', (state) => {
            console.log('🎵 Player state changed:', state);
            if (!state) return;
            
            this.updatePlayerUI(state);
            this.updatePlayingCards(state.track_window.current_track);
        });
        
        // Ready
        player.addListener('ready', ({ device_id }) => {
            console.log('✅ Player ready with Device ID:', device_id);
            this.deviceId = device_id;
            this.spotifyPlayer = player;
            this.isPlayerReady = true;
        });
        
        // Not Ready
        player.addListener('not_ready', ({ device_id }) => {
            console.log('❌ Device ID has gone offline:', device_id);
            this.isPlayerReady = false;
        });
        
        // Connect to the player!
        console.log('🔗 Connecting to Spotify...');
        player.connect().then(success => {
            if (success) {
                console.log('✅ Successfully connected to Spotify Web Playback SDK');
            } else {
                console.error('❌ Failed to connect to Spotify Web Playback SDK');
            }
        });
    }
    
    async playSong(song) {
        console.log('🎵 playSong called with:', song);
        console.log('🎵 Player ready?', this.isPlayerReady);
        console.log('🎵 Access token?', this.accessToken ? 'Yes' : 'No');
        console.log('🎵 Device ID?', this.deviceId);
        console.log('🎵 Spotify ID?', song.spotify_id);
        
        if (!this.isPlayerReady || !song.spotify_id) {
            console.log('❌ Player not ready or no Spotify ID');
            console.log('   - Player ready:', this.isPlayerReady);
            console.log('   - Spotify ID:', song.spotify_id);
            
            if (!this.isPlayerReady) {
                console.log('🔄 Attempting to reinitialize player...');
                this.initSpotifyPlayer();
            }
            return;
        }
        
        console.log('✅ Starting playback for:', song.song, 'by', song.artist);
        
        try {
            // Transfer playback to our device first if needed
            await this.ensureDeviceActive();
            
            console.log('🎵 Playing track with URI:', `spotify:track:${song.spotify_id}`);
            
            // Play the track
            const response = await fetch(`https://api.spotify.com/v1/me/player/play?device_id=${this.deviceId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.accessToken}`
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
            
            console.log('✅ Playback started successfully');
            this.currentTrack = song;
            
        } catch (error) {
            console.error('❌ Error playing song:', error);
            
            // Show user-friendly error message
            alert(`Unable to play "${song.song}" by ${song.artist}. ${error.message || 'Please make sure Spotify is running and you have Premium.'}`);
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
        if (!this.spotifyPlayer) return;
        
        try {
            await this.spotifyPlayer.togglePlay();
        } catch (error) {
            console.error('Error toggling playback:', error);
        }
    }
    
    async previousTrack() {
        if (!this.spotifyPlayer) return;
        
        try {
            await this.spotifyPlayer.previousTrack();
        } catch (error) {
            console.error('Error going to previous track:', error);
        }
    }
    
    async nextTrack() {
        if (!this.spotifyPlayer) return;
        
        try {
            await this.spotifyPlayer.nextTrack();
        } catch (error) {
            console.error('Error going to next track:', error);
        }
    }
    
    async seekToPosition(event) {
        if (!this.spotifyPlayer) return;
        
        const progressBar = event.currentTarget;
        const rect = progressBar.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const progressBarWidth = rect.width;
        const percentage = clickX / progressBarWidth;
        
        try {
            const state = await this.spotifyPlayer.getCurrentState();
            if (state) {
                const positionMs = Math.floor(state.duration * percentage);
                await this.spotifyPlayer.seek(positionMs);
            }
        } catch (error) {
            console.error('Error seeking:', error);
        }
    }
    
    updatePlayerUI(state) {
        const track = state.track_window.current_track;
        const isPaused = state.paused;
        
        // Update track info
        const coverUrl = track.album?.images?.[0]?.url || '';
        document.getElementById('player-cover').src = coverUrl;
        document.getElementById('player-title').textContent = track.name || 'Unknown Track';
        document.getElementById('player-artist').textContent = track.artists?.map(artist => artist.name).join(', ') || 'Unknown Artist';
        
        // Update play button
        const playBtn = document.getElementById('play-btn');
        playBtn.textContent = isPaused ? '▶' : '⏸';
        
        // Update progress
        this.updateProgress(state.position, state.duration);
    }
    
    updateProgress(position, duration) {
        const progressFilled = document.getElementById('progress-filled');
        const currentTime = document.getElementById('current-time');
        const totalTime = document.getElementById('total-time');
        
        const progressPercent = duration > 0 ? (position / duration) * 100 : 0;
        progressFilled.style.width = `${progressPercent}%`;
        
        currentTime.textContent = this.formatTime(position);
        totalTime.textContent = this.formatTime(duration);
    }
    
    updatePlayingCards(currentTrack) {
        if (!currentTrack) return;
        
        const currentSpotifyId = currentTrack.id;
        
        document.querySelectorAll('.song-card').forEach(card => {
            if (card.dataset.spotifyId === currentSpotifyId) {
                card.classList.add('playing');
            } else {
                card.classList.remove('playing');
            }
        });
    }
    
    formatTime(ms) {
        const minutes = Math.floor(ms / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new IntelligentSearchApp();
});

// Handle Spotify Web Playback SDK ready callback
window.onSpotifyWebPlaybackSDKReady = () => {
    console.log('Spotify Web Playback SDK is ready');
    if (window.app && window.app.accessToken) {
        window.app.initSpotifyPlayer();
    }
}; 