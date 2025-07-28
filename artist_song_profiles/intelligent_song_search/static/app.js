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
        
        // Auto-play queue management
        this.currentSongIndex = -1;  // Index in search results
        this.isAutoPlayEnabled = true;  // Could make this configurable later
        this.searchResultsId = null;  // To detect result changes
        this.isManualSkip = false;  // Track if user manually skipped
        this.lastTrackId = null;  // Track last played track for auto-advance detection
        this.autoAdvancePending = false;  // Prevent multiple rapid auto-advances
        this.lastAutoAdvanceTime = 0;  // Rate limiting for auto-advances (0 = no previous auto-advance)
        this.isPlayingSong = false;  // Track if we're currently starting a song
        this.autoPlayCheckInterval = null;  // Backup auto-play checker
        this.lastProcessedTrackEnd = null;  // Prevent duplicate track-end processing
        
        this.init();
    }
    
    init() {
        this.bindEventListeners();
        this.checkAuthStatus();
        this.initSpotifyPlayer();
        this.updateAutoPlayUI();  // Initialize auto-play button state
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
        
        // Auto-play toggle
        document.getElementById('autoplay-toggle').addEventListener('click', () => {
            this.toggleAutoPlay();
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
        
        // Create a search identifier based on actual search parameters
        const newSearchId = `${searchType}:${embedType}:${query}:${this.currentQuerySong?.song_idx || ''}`;
        const isNewSearch = this.searchResultsId !== newSearchId;
        
        // Reset pagination for new searches
        this.currentOffset = 0;
        this.searchResults = [];
        
        // Reset queue when starting a new search
        if (isNewSearch) {
            console.log('🎵 New search detected, resetting queue');
            this.searchResultsId = newSearchId;
            this.resetAutoPlayQueue();
        }
        
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
            const previousResultsCount = this.searchResults.length;
            this.searchResults = [...this.searchResults, ...data.results];
            this.currentOffset = data.pagination.offset + data.pagination.limit;
            this.hasMoreResults = data.pagination.has_more;
            this.displayResults(data, true);
            
            console.log('🎵 Loaded more results, queue now has', this.searchResults.length, 'songs');
            console.log('🎵 Added', data.results.length, 'new songs to queue (was', previousResultsCount, ')');
            
            // If user was at the end of queue and new results loaded, they can now continue
            if (this.currentSongIndex === previousResultsCount - 1 && data.results.length > 0) {
                console.log('🎵 Queue extended - auto-play can now continue from end of previous results');
            }
            
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
        
        // Reset auto-play queue
        this.resetAutoPlayQueue();
    }
    
    // Reset auto-play queue state
    resetAutoPlayQueue() {
        console.log('🎵 Resetting auto-play queue');
        this.currentSongIndex = -1;
        this.lastTrackId = null;
        this.isManualSkip = false;
        this.autoAdvancePending = false;
        this.lastAutoAdvanceTime = 0;
        this.isPlayingSong = false;
        this.lastProcessedTrackEnd = null;
        
        // Stop current playback when search changes (if playing from our queue)
        if (this.spotifyPlayer && this.currentTrack) {
            console.log('🎵 Stopping current playback due to search change');
            this.spotifyPlayer.pause().catch(error => {
                console.error('Error pausing during queue reset:', error);
            });
        }
    }
    
    // Backup auto-play checker - runs periodically to catch missed track endings
    startAutoPlayChecker() {
        if (this.autoPlayCheckInterval) {
            clearInterval(this.autoPlayCheckInterval);
        }
        
        this.autoPlayCheckInterval = setInterval(async () => {
            if (!this.isPlayerReady || !this.spotifyPlayer || !this.isAutoPlayEnabled || this.currentSongIndex < 0) {
                return;
            }
            
            try {
                const state = await this.spotifyPlayer.getCurrentState();
                if (state) {
                    // Less verbose logging for backup checks
                    this.handleAutoPlayCheck(state, true); // Pass flag to indicate backup check
                }
            } catch (error) {
                console.error('Error in backup auto-play check:', error);
            }
        }, 2000); // Check every 2 seconds to be less aggressive
        
        console.log('🎵 Started backup auto-play checker');
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
            
            // Debug logging for auto-play troubleshooting
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
            console.log('✅ Player ready with Device ID:', device_id);
            this.deviceId = device_id;
            this.spotifyPlayer = player;
            this.isPlayerReady = true;
            
            // Start backup auto-play checker (disabled for now to avoid interference)
            // this.startAutoPlayChecker();
        });
        
        // Not Ready
        player.addListener('not_ready', ({ device_id }) => {
            console.log('❌ Device ID has gone offline:', device_id);
            this.isPlayerReady = false;
            
            // Stop backup checker when player goes offline
            if (this.autoPlayCheckInterval) {
                clearInterval(this.autoPlayCheckInterval);
                this.autoPlayCheckInterval = null;
            }
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
    
    async playSong(song, isAutoAdvance = false) {
        console.log('🎵 playSong called with:', song, 'isAutoAdvance:', isAutoAdvance);
        
        // Prevent concurrent playSong calls
        if (this.isPlayingSong) {
            console.log('🎵 Already playing a song, ignoring request');
            return;
        }
        
        // Rate limiting only for auto-advances, not manual song clicks
        // Be very lenient for auto-advances to allow natural track progression
        const now = Date.now();
        const timeSinceLastAdvance = now - this.lastAutoAdvanceTime;
        
        if (isAutoAdvance && this.lastAutoAdvanceTime > 0 && timeSinceLastAdvance < 100) {
            console.log('🎵 Rate limiting auto-advance - too soon since last play request');
            console.log('🎵 Time since last advance:', timeSinceLastAdvance, 'ms');
            return;
        }
        
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
        
        this.isPlayingSong = true;
        if (isAutoAdvance) {
            this.lastAutoAdvanceTime = now;
            // Reset the auto-advance pending flag since we're now actually starting the song
            this.autoAdvancePending = false;
            console.log('🎵 Auto-advance: Reset autoAdvancePending flag, set lastAutoAdvanceTime');
        }
        console.log('✅ Starting playback for:', song.song, 'by', song.artist);
        
        try {
            // Transfer playback to our device first if needed
            await this.ensureDeviceActive();
            
            console.log('🎵 Playing track with URI:', `spotify:track:${song.spotify_id}`);
            
            // Find and store the index of this song in current results for queue management
            this.currentSongIndex = this.searchResults.findIndex(result => 
                result.spotify_id === song.spotify_id
            );
            console.log('🎵 Current song index in results:', this.currentSongIndex);
            
            // If song not found in current results, it might be from a previous search
            if (this.currentSongIndex === -1) {
                console.log('🎵 Song not found in current results - might be from previous search');
            }
            
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
            this.lastTrackId = song.spotify_id;  // Track for auto-advance detection
            
            // Reset the playing flag after successful start
            // The song is now playing, so we're no longer "starting" it
            setTimeout(() => {
                this.isPlayingSong = false;
                this.lastProcessedTrackEnd = null; // Reset for new song
                // Reset manual skip flag since a new song is now playing successfully
                if (this.isManualSkip) {
                    console.log('🎵 Resetting manual skip flag - new song playing successfully');
                    this.isManualSkip = false;
                }
                console.log('🎵 Reset isPlayingSong flag and trackEnd processing after successful playback start');
            }, 1000); // Give it a second to start playing
            
        } catch (error) {
            console.error('❌ Error playing song:', error);
            
            // Reset current song index on error to prevent queue issues
            this.currentSongIndex = -1;
            
            // Show user-friendly error message
            console.log(`Unable to play "${song.song}" by ${song.artist}. ${error.message || 'Please make sure Spotify is running and you have Premium.'}`);
            
            // Reset flags on error
            this.isPlayingSong = false;
            if (isAutoAdvance) {
                this.autoAdvancePending = false;
                console.log('🎵 Reset autoAdvancePending flag due to error during auto-advance');
            }
            // Also reset manual skip flag on error to prevent getting stuck
            if (this.isManualSkip) {
                console.log('🎵 Resetting manual skip flag due to playback error');
                this.isManualSkip = false;
            }
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
        
        // Rate limiting for play/pause button
        const now = Date.now();
        if (now - this.lastAutoAdvanceTime < 500) {
            console.log('🎵 Toggle playback too soon, ignoring');
            return;
        }
        
        try {
            // Mark as manual interaction to prevent auto-play conflicts
            this.isManualSkip = true;
            this.autoAdvancePending = false; // Reset any pending auto-advance
            // Don't set lastAutoAdvanceTime for manual interactions - only for actual auto-advances
            await this.spotifyPlayer.togglePlay();
            
            // Reset manual skip flag after a short delay
            setTimeout(() => {
                this.isManualSkip = false;
            }, 1000);
        } catch (error) {
            console.error('Error toggling playback:', error);
        }
    }
    
    async previousTrack() {
        console.log('🎵 previousTrack called - using search results queue');
        
        // Rate limiting for manual clicks
        const now = Date.now();
        if (now - this.lastAutoAdvanceTime < 1000) {
            console.log('🎵 Manual previous track too soon, ignoring');
            return;
        }
        
        this.isManualSkip = true;  // Mark as manual skip for consistency
        this.autoAdvancePending = false; // Reset any pending auto-advance
        await this.playPreviousInResults();
    }
    
    async nextTrack() {
        console.log('🎵 nextTrack called - using search results queue');
        
        // Rate limiting for manual clicks
        const now = Date.now();
        if (now - this.lastAutoAdvanceTime < 1000) {
            console.log('🎵 Manual next track too soon, ignoring');
            return;
        }
        
        this.isManualSkip = true;  // Mark as manual skip
        this.autoAdvancePending = false; // Reset any pending auto-advance
        await this.playNextInResults();
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
    
    // Auto-play queue management methods
    handleAutoPlayCheck(state, isBackupCheck = false) {
        // Basic checks - exit early if conditions not met
        if (!state || !this.isAutoPlayEnabled || this.currentSongIndex < 0 || this.isPlayingSong || this.autoAdvancePending) {
            return;
        }
        
        const currentTrack = state.track_window.current_track;
        if (!currentTrack) return;
        
        // Debug: Show current track info
        if (!isBackupCheck) {
            console.log('🎵 Track info:', {
                currentTrackId: currentTrack.id,
                lastTrackId: this.lastTrackId,
                trackChanged: currentTrack.id !== this.lastTrackId,
                manualSkipFlag: this.isManualSkip
            });
        }
        
        // Rate limiting - don't check too frequently (but be less aggressive)
        const now = Date.now();
        if (now - this.lastAutoAdvanceTime < 1000) { // More lenient 1 second
            return;
        }
        
        // Multiple ways to detect track ending - be more flexible
        const position = state.position || 0;
        const duration = state.duration || 0;
        
        // Method 1: Traditional end-of-track detection (more lenient)
        const isNearEnd = state.paused && 
                         duration > 10000 && // Only for tracks longer than 10 seconds (reduced)
                         position >= duration - 2000 && // Within 2 seconds of end (more lenient)
                         position > duration * 0.8; // Must have played at least 80% (more lenient)
        
        // Method 2: Position equals duration (exact end)
        const isAtExactEnd = state.paused && position >= duration && duration > 0;
        
        // Method 3: Position very close to duration percentage-wise
        const isAtPercentageEnd = state.paused && 
                                 duration > 0 && 
                                 (position / duration) >= 0.98; // 98% through
        
        // Method 4: Spotify sometimes reports position as 0 when track ends
        // But make sure this isn't the very start of the track by checking we had a valid lastTrackId
        const isPausedAtZero = state.paused && 
                              position === 0 && 
                              this.lastTrackId === currentTrack.id && 
                              duration > 0 &&
                              !state.loading; // Don't trigger on loading states
        
        const shouldAutoAdvance = (isNearEnd || isAtExactEnd || isAtPercentageEnd || isPausedAtZero) && 
                                 !this.isManualSkip && 
                                 !state.loading;
        
        // Create a unique identifier for this track ending to prevent duplicate processing
        // Use a longer time window to prevent rapid repeats
        const trackEndId = `${currentTrack.id}_${Math.floor(now / 3000)}`; // Unique per track per 3 seconds
        
        if (!isBackupCheck) {
            console.log('🎵 Auto-advance check:', {
                isNearEnd, isAtExactEnd, isAtPercentageEnd, isPausedAtZero,
                shouldAutoAdvance, position, duration, paused: state.paused,
                manualSkip: this.isManualSkip, loading: state.loading,
                trackEndId, lastProcessed: this.lastProcessedTrackEnd,
                blockingReason: shouldAutoAdvance ? null : (
                    this.isManualSkip ? 'manual skip flag' : 
                    state.loading ? 'still loading' :
                    !(isNearEnd || isAtExactEnd || isAtPercentageEnd || isPausedAtZero) ? 'no end condition met' : 'unknown'
                )
            });
        }
        
        if (shouldAutoAdvance) {
            // Prevent duplicate processing of the same track ending
            if (this.lastProcessedTrackEnd === trackEndId) {
                if (!isBackupCheck) {
                    console.log('🎵 Already processed this track ending, skipping');
                }
                return;
            }
            
            console.log('🎵 ✅ Track ended naturally - advancing to next');
            console.log('🎵 Trigger reason:', {
                isNearEnd, isAtExactEnd, isAtPercentageEnd, isPausedAtZero
            });
            console.log('🎵 Current song index:', this.currentSongIndex);
            console.log('🎵 Total results:', this.searchResults.length);
            
            // Mark this track ending as processed
            this.lastProcessedTrackEnd = trackEndId;
            
            // Set pending flag and advance immediately (no setTimeout wrapper)
            this.autoAdvancePending = true;
            // Don't set lastAutoAdvanceTime here - let playSong set it when actually starting
            
            // Call directly without setTimeout to avoid race condition
            console.log('🎵 About to call playNextInResults with autoAdvancePending:', this.autoAdvancePending);
            this.playNextInResults(true);
        }
        
        // Update last track ID
        if (currentTrack.id !== this.lastTrackId) {
            console.log('🎵 Track changed from', this.lastTrackId, 'to', currentTrack.id);
            console.log('🎵 Manual skip flag before track change:', this.isManualSkip);
            this.lastTrackId = currentTrack.id;
            
            // Reset processing tracking for new track
            this.lastProcessedTrackEnd = null;
            
            // Reset manual skip flag after track changes, with delay
            // NOTE: This is backup logic - the main reset happens in playSong timeout
            if (this.isManualSkip) {
                setTimeout(() => {
                    console.log('🎵 Backup: Resetting manual skip flag via track change detection');
                    this.isManualSkip = false;
                }, 2000); // Longer delay as backup
            }
        }
    }
    
    async playNextInResults(isAutoAdvance = false) {
        console.log('🎵 playNextInResults called, isAutoAdvance:', isAutoAdvance);
        console.log('🎵 Current index:', this.currentSongIndex);
        console.log('🎵 Total results:', this.searchResults.length);
        console.log('🎵 All flags:', {
            isPlayingSong: this.isPlayingSong,
            autoAdvancePending: this.autoAdvancePending,
            isAutoPlayEnabled: this.isAutoPlayEnabled,
            timeSinceLastAdvance: Date.now() - this.lastAutoAdvanceTime,
            lastAutoAdvanceTime: this.lastAutoAdvanceTime
        });
        
        // Prevent concurrent calls, but allow auto-advance to proceed
        if (this.isPlayingSong || (this.autoAdvancePending && !isAutoAdvance)) {
            console.log('🎵 Already advancing or playing, ignoring call');
            console.log('🎵 Flags - isPlayingSong:', this.isPlayingSong, 'autoAdvancePending:', this.autoAdvancePending, 'isAutoAdvance:', isAutoAdvance);
            return;
        }
        
        if (this.currentSongIndex >= 0 && 
            this.currentSongIndex < this.searchResults.length - 1) {
            
            const nextSong = this.searchResults[this.currentSongIndex + 1];
            console.log('🎵 Playing next song:', nextSong.song, 'by', nextSong.artist);
            
            // Pass the isAutoAdvance flag to playSong
            await this.playSong(nextSong, isAutoAdvance);
            
        } else if (this.currentSongIndex === this.searchResults.length - 1) {
            // End of queue
            console.log('🎵 Reached end of search results');
            
            if (this.hasMoreResults) {
                console.log('🎵 More results available, but not auto-loading for now');
            }
            
            // Reset auto-advance flag at end of queue
            if (isAutoAdvance) {
                this.autoAdvancePending = false;
                console.log('🎵 Reset autoAdvancePending flag - reached end of queue');
            }
            
            // Stop playback gracefully
            try {
                if (this.spotifyPlayer) {
                    await this.spotifyPlayer.pause();
                }
            } catch (error) {
                console.error('Error pausing at end of queue:', error);
            }
        } else {
            // Invalid state - reset flags
            console.log('🎵 Invalid queue state, resetting flags');
            if (isAutoAdvance) {
                this.autoAdvancePending = false;
            }
        }
    }
    
    async playPreviousInResults(isAutoAdvance = false) {
        console.log('🎵 playPreviousInResults called, isAutoAdvance:', isAutoAdvance);
        console.log('🎵 Current index:', this.currentSongIndex);
        
        if (this.currentSongIndex > 0) {
            const prevSong = this.searchResults[this.currentSongIndex - 1];
            console.log('🎵 Playing previous song:', prevSong.song, 'by', prevSong.artist);
            await this.playSong(prevSong, isAutoAdvance);
        } else {
            console.log('🎵 Already at beginning of search results');
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
    
    toggleAutoPlay() {
        this.isAutoPlayEnabled = !this.isAutoPlayEnabled;
        console.log('🎵 Auto-play toggled:', this.isAutoPlayEnabled ? 'enabled' : 'disabled');
        this.updateAutoPlayUI();
    }
    
    updateAutoPlayUI() {
        const autoPlayBtn = document.getElementById('autoplay-toggle');
        if (this.isAutoPlayEnabled) {
            autoPlayBtn.classList.add('active');
            autoPlayBtn.title = 'Auto-play enabled - Click to disable';
            autoPlayBtn.style.opacity = '1';
        } else {
            autoPlayBtn.classList.remove('active');
            autoPlayBtn.title = 'Auto-play disabled - Click to enable';
            autoPlayBtn.style.opacity = '0.5';
        }
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