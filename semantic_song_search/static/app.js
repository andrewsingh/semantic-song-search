// Semantic Song Search App JavaScript

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class SemanticSearchApp {
    constructor() {
        this.currentQuery = null;
        this.currentQuerySong = null;
        this.currentSearchType = 'text'; // Initialize to match HTML default
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
        this.playerInitPromise = null; // Promise for player initialization
        
        // Top artists filter
        this.topArtists = [];
        this.topArtistsLoaded = false;
        this.isAuthenticated = false;
        this.isFiltered = false;
        this.originalSearchResults = []; // Store unfiltered results for client-side filtering
        
        // Manual song selection
        this.isManualSelectionMode = false;
        this.selectedSongs = new Set(); // Set of song indices that are selected
        
        // Auto-play queue management
        this.currentSongIndex = -1;  // Index in search results
        this.isAutoPlayEnabled = true;  // Auto-play always enabled
        this.searchResultsId = null;  // To detect result changes
        this.isManualSkip = false;  // Track if user manually skipped
        this.lastTrackId = null;  // Track last played track for auto-advance detection
        this.autoAdvancePending = false;  // Prevent multiple rapid auto-advances
        this.lastAutoAdvanceTime = 0;  // Rate limiting for auto-advances (0 = no previous auto-advance)
        this.isPlayingSong = false;  // Track if we're currently starting a song
        this.autoPlayCheckInterval = null;  // Backup auto-play checker
        this.lastProcessedTrackEnd = null;  // Prevent duplicate track-end processing
        
        // Analytics tracking
        this.sessionStartTime = Date.now();
        this.searchCount = 0;
        this.songsPlayed = 0;
        this.playlistsCreated = 0;
        
        this.init();
    }
    
    init() {
        this.bindEventListeners();
        this.checkAuthStatus();
        
        // Ensure currentSearchType is synced with the initial HTML state
        this.currentSearchType = this.getSearchType();
        
        // Don't initialize player here - it will be initialized when needed in playSong()
        // this.initSpotifyPlayer();
        
        // Track initial page load
        this.trackPageLoad();
    }
    
    trackPageLoad() {
        if (typeof mixpanel !== 'undefined') {
            // Get timezone and language information
            const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            const language = navigator.language || navigator.userLanguage;
            const languages = navigator.languages || [language];
            
            mixpanel.track('Page Loaded', {
                'page_title': document.title,
                'url': window.location.href,
                'referrer': document.referrer,
                'user_agent': navigator.userAgent,
                'screen_width': window.screen.width,
                'screen_height': window.screen.height,
                'viewport_width': window.innerWidth,
                'viewport_height': window.innerHeight,
                'timezone': timezone,
                'language': language,
                'languages': languages.join(', '),
                'platform': navigator.platform,
                'cookie_enabled': navigator.cookieEnabled
            });
        }
    }
    
    // Helper function to get current search type from segmented control
    getSearchType() {
        try {
            const checkedRadio = document.querySelector('input[name="search-type"]:checked');
            return checkedRadio ? checkedRadio.value : 'text';
        } catch (error) {
            console.warn('Error getting search type from segmented control:', error);
            return 'text'; // Safe fallback
        }
    }
    
    bindEventListeners() {
        // Search type change - segmented control
        const searchTypeRadios = document.querySelectorAll('input[name="search-type"]');
        if (searchTypeRadios.length > 0) {
            searchTypeRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        this.handleSearchTypeChange(e.target.value);
                    }
                });
            });
        } else {
            console.warn('Search type radio buttons not found in DOM');
        }
        
        // Embedding type change - auto-rerun search if results exist
        document.getElementById('embed-type').addEventListener('change', (e) => {
            this.handleEmbedTypeChange(e.target.value);
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
        
        // Login button
        document.getElementById('login-btn').addEventListener('click', () => {
            window.location.href = '/login';
        });
        
        // Logout button
        document.getElementById('logout-btn').addEventListener('click', () => {
            window.location.href = '/logout';
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
        
        // Export accordion toggle
        document.getElementById('export-accordion-btn').addEventListener('click', () => {
            this.toggleExportAccordion();
        });
        
        // Export button
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportToPlaylist();
        });
        
        // Manual selection toggle
        document.getElementById('manual-selection-toggle').addEventListener('change', (e) => {
            this.toggleManualSelection(e.target.checked);
        });
        
        // Top artists filter checkbox
        document.getElementById('top-artists-filter').addEventListener('change', async (e) => {
            console.log(`🎛️ Top artists filter checkbox changed: ${e.target.checked ? 'CHECKED' : 'UNCHECKED'}`);
            console.log(`🎛️ Current state - topArtistsLoaded: ${this.topArtistsLoaded}, isAuthenticated: ${this.isAuthenticated}, top artists count: ${this.topArtists.length}`);
            
            // Track filter toggle
            if (typeof mixpanel !== 'undefined') {
                mixpanel.track('Top Artists Filter Toggled', {
                    'filter_enabled': e.target.checked,
                    'is_authenticated': this.isAuthenticated,
                    'top_artists_loaded': this.topArtistsLoaded,
                    'top_artists_count': this.topArtists.length,
                    'has_existing_results': this.searchResults.length > 0,
                    'results_count': this.searchResults.length
                });
            }
            
            // If checked and we don't have top artists yet, load them
            if (e.target.checked && !this.topArtistsLoaded && this.isAuthenticated) {
                console.log(`🎛️ Loading top artists because checkbox was checked but not loaded yet...`);
                await this.loadTopArtists();
            }
            
            // If we have existing search results, filter them client-side instead of re-running the search
            if (this.searchResults.length > 0) {
                console.log(`🔄 Filter toggled to ${e.target.checked ? 'ON' : 'OFF'}, filtering ${this.searchResults.length} existing results client-side...`);
                this.applyClientSideFilter();
            } else {
                console.log(`🔄 No existing results to filter - searchResults.length: ${this.searchResults.length}`);
            }
        });
        
        // Load more button
        document.getElementById('load-more-btn').addEventListener('click', () => {
            this.loadMoreResults();
        });
        
        // Track when user leaves the page
        window.addEventListener('beforeunload', () => {
            const sessionDuration = Math.round((Date.now() - this.sessionStartTime) / 1000);
            if (typeof mixpanel !== 'undefined') {
                mixpanel.track('Session Ended', {
                    'session_duration_seconds': sessionDuration,
                    'searches_performed': this.searchCount || 0,
                    'songs_played': this.songsPlayed || 0,
                    'playlists_created': this.playlistsCreated || 0
                });
            }
        });

    }
    
    handleSearchTypeChange(searchType) {
        const suggestionsContainer = document.getElementById('suggestions');
        const querySection = document.getElementById('query-section');
        const searchInput = document.getElementById('search-input');
        
        // Track search type change
        if (typeof mixpanel !== 'undefined') {
            mixpanel.track('Search Type Changed', {
                'new_search_type': searchType,
                'previous_search_type': this.currentSearchType || 'unknown',
                'has_active_search': this.searchResults.length > 0
            });
        }
        
        // Update current search type immediately
        this.currentSearchType = searchType;
        
        if (searchType === 'song') {
            searchInput.placeholder = "🔍 Search for a song or artist... (e.g., \"Espresso\", \"Sabrina Carpenter\")";
            this.clearResults();
        } else {
            searchInput.placeholder = "🔍 Describe the vibe you're looking for... (e.g., \"upbeat summery pop\", \"motivational workout hip hop\")";
            suggestionsContainer.style.display = 'none';
            querySection.style.display = 'none';
            this.clearResults();
        }
    }
    
    handleEmbedTypeChange(embedType) {
        console.log(`🎛️ Embedding type changed to: ${embedType}`);
        
        // Track embed type change
        if (typeof mixpanel !== 'undefined') {
            mixpanel.track('Embed Type Changed', {
                'new_embed_type': embedType,
                'previous_embed_type': this.currentEmbedType || 'unknown',
                'has_active_search': this.searchResults.length > 0
            });
        }
        
        this.currentEmbedType = embedType;
        
        // Don't auto-rerun if we're currently loading more results
        if (this.isLoadingMore) {
            console.log(`🔄 Not auto-rerunning search - currently loading more results`);
            return;
        }
        
        // Check if we have existing search results and a query to re-run
        const query = document.getElementById('search-input').value.trim();
        const hasResults = this.searchResults.length > 0;
        const hasQuery = query.length > 0;
        
        // For song-to-song searches, also check if we have a selected query song
        const searchType = this.getSearchType();
        const hasValidQuery = hasQuery || (searchType === 'song' && this.currentQuerySong);
        
        if (hasResults && hasValidQuery) {
            console.log(`🔄 Auto-rerunning search with new embedding type: ${embedType}`);
            // Auto-rerun the search with the new embedding type
            this.handleSearch();
        } else {
            console.log(`🔄 Not auto-rerunning search - hasResults: ${hasResults}, hasValidQuery: ${hasValidQuery}, searchType: ${searchType}`);
        }
    }
    
    async handleSearchInput(query) {
        const searchType = this.getSearchType();
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
        const searchType = this.getSearchType();
        const embedType = document.getElementById('embed-type').value;
        const query = document.getElementById('search-input').value.trim();
        const topArtistsFilter = document.getElementById('top-artists-filter');
        let filterTopArtists = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        if (!query) return;
        
        // Track search initiation on frontend with comprehensive context
        if (typeof mixpanel !== 'undefined') {
            const searchProperties = {
                'search_type': searchType,
                'embed_type': embedType,
                'is_filtered': filterTopArtists,
                'is_manual_selection': this.isManualSelectionMode,
                'selected_songs_count': this.selectedSongs.size,
                'has_spotify_auth': this.isAuthenticated,
                'has_results': this.searchResults.length > 0,
                'is_new_search': this.searchResultsId !== `${searchType}:${embedType}:${query}:${this.currentQuerySong?.song_idx || ''}`
            };
            
            // Add query-specific information
            if (searchType === 'text' && query) {
                searchProperties.query = query;
                searchProperties.query_length = query.length;
            } else if (searchType === 'song' && this.currentQuerySong) {
                searchProperties.has_query_song = true;
                searchProperties.query_song_idx = this.currentQuerySong.song_idx;
                searchProperties.query_song_name = this.currentQuerySong.song || '';
                searchProperties.query_artist_name = this.currentQuerySong.artist || '';
            }
            
            mixpanel.track('Search Initiated', searchProperties);
        }
        
        this.searchCount++;
        
        // If top artists filter is enabled but not loaded, load them first
        if (filterTopArtists && !this.topArtistsLoaded) {
            await this.loadTopArtists();
            // If loading failed, uncheck the filter and continue without it
            if (!this.topArtistsLoaded) {
                topArtistsFilter.checked = false;
                filterTopArtists = false; // Update the variable to reflect the unchecked state
            }
        }
        
        // Create a search identifier based on actual search parameters (excluding filter since it's now client-side)
        const newSearchId = `${searchType}:${embedType}:${query}:${this.currentQuerySong?.song_idx || ''}`;
        const isNewSearch = this.searchResultsId !== newSearchId;
        
        // Reset pagination for new searches
        this.currentOffset = 0;
        this.searchResults = [];
        this.originalSearchResults = [];
        
        // Reset manual selection for new searches
        this.isManualSelectionMode = false;
        this.selectedSongs.clear();
        const manualSelectionToggle = document.getElementById('manual-selection-toggle');
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        // Update export form to show number input instead of selection info
        this.updateExportFormDisplay();
        
        // Reset queue when starting a new search
        if (isNewSearch) {
            console.log('🎵 New search detected, resetting queue');
            this.searchResultsId = newSearchId;
            this.resetAutoPlayQueue();
        }
        
        // Store current search parameters for playlist name auto-population
        this.currentQuery = query;
        this.currentSearchType = searchType;
        
        this.showLoading(true);
        this.hideWelcomeMessage();
        
        try {
            const requestData = {
                search_type: searchType,
                embed_type: embedType,
                query: query,
                k: 20,
                offset: 0
                // Note: No longer sending filter_top_artists since we do client-side filtering
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
            this.originalSearchResults = [...data.results]; // Store original unfiltered results for client-side filtering
            this.currentOffset = data.pagination.offset + data.pagination.limit;
            this.totalResultsCount = data.pagination.total_count;
            this.hasMoreResults = data.pagination.has_more;
            this.isFiltered = false; // Reset since server doesn't filter anymore
            
            // Apply client-side filter if checkbox is currently checked
            const topArtistsFilter = document.getElementById('top-artists-filter');
            if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.topArtistsLoaded) {
                console.log('🔍 Auto-applying client-side filter after search because checkbox is checked');
                this.applyClientSideFilter();
            } else {
                this.displayResults(data, false);
            }
            
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
        
        // Track load more click
        if (typeof mixpanel !== 'undefined') {
            mixpanel.track('Load More Clicked', {
                'current_results_count': this.searchResults.length,
                'current_offset': this.currentOffset,
                'search_type': this.currentSearchData.search_type,
                'embed_type': this.currentSearchData.embed_type,
                'has_filter_active': document.getElementById('top-artists-filter').checked,
                'is_manual_selection_mode': this.isManualSelectionMode
            });
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
            
            // Add new results to original unfiltered results
            this.originalSearchResults = [...this.originalSearchResults, ...data.results];
            this.currentOffset = data.pagination.offset + data.pagination.limit;
            this.hasMoreResults = data.pagination.has_more;
            
            // Auto-select new songs if manual selection is enabled
            if (this.isManualSelectionMode) {
                data.results.forEach(song => {
                    this.selectedSongs.add(song.song_idx);
                });
                console.log(`🎯 Auto-selected ${data.results.length} new songs for manual selection mode`);
            }
            
            // Check if we have client-side filtering active
            const topArtistsFilter = document.getElementById('top-artists-filter');
            if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.topArtistsLoaded) {
                console.log('🔍 Re-applying client-side filter after loading more results');
                this.applyClientSideFilter();
            } else {
                // No filtering - just add new results and display
                this.searchResults = [...this.searchResults, ...data.results];
                this.displayResults(data, true);
                
                // Update manual selection state for newly loaded cards if needed
                if (this.isManualSelectionMode) {
                    this.updateAllCardSelections();
                    this.updateExportFormDisplay(); // Update export form with new selection count
                }
            }
            
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
        
        // Update header (show current results count with new formatting)
        this.updateResultsCount();
        
        searchInfo.textContent = `${data.search_type} search • ${data.embed_type.replace('_', ' ')}`;
        resultsHeader.style.display = 'flex';
        
        // Clear results grid only for new searches, not for load more
        if (!isLoadMore) {
            resultsGrid.innerHTML = '';
            // Reset listener tracking for new searches
            this.resetEventListenerTracking();
        }
        
        // Handle empty results case
        if (!isLoadMore && this.searchResults.length === 0) {
            const message = this.isFiltered ? 
                'No songs found matching your filters. Try searching without filters or adding more listening history to Spotify.' :
                'No results found. Try adjusting your search terms or selecting a different embedding type.';
            
            resultsGrid.innerHTML = `
                <div class="no-results-message">
                    <h3>No results found</h3>
                    <p>${message}</p>
                </div>
            `;
            loadMoreContainer.style.display = 'none';
            document.getElementById('export-section').style.display = 'none';
            return;
        }
        
        // Show export section when results are available (only for new searches)
        if (!isLoadMore && this.searchResults.length > 0) {
            document.getElementById('export-section').style.display = 'block';
            // Update the song count input placeholder to show available results
            this.updateSongCountHint();
        }
        
        // Add new results to the grid  
        const startIndex = isLoadMore ? this.searchResults.length - data.results.length : 0;
        data.results.forEach((song, index) => {
            const card = document.createElement('div');
            let cardClasses = 'song-card';
            
            // Add selection-related classes
            if (this.isManualSelectionMode) {
                cardClasses += ' selectable';
                if (this.selectedSongs.has(song.song_idx)) {
                    cardClasses += ' selected';
                }
            }
            
            card.className = cardClasses;
            card.dataset.spotifyId = song.spotify_id; // Set the spotify ID for the updatePlayingCards function
            card.innerHTML = this.createSongCardHTML(song, { 
                rank: startIndex + index + 1, 
                similarity: song.similarity,
                fieldValue: song.field_value,
                embedType: data.embed_type,
                isSelected: this.selectedSongs.has(song.song_idx)
            });
            
            // Don't add the old click listener here - we'll handle it in attachSongCardEventListeners
            
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
        
        // Ensure results container has correct CSS class for manual selection mode
        const resultsContainer = document.getElementById('results-container');
        if (this.isManualSelectionMode) {
            resultsContainer.classList.add('manual-selection-mode');
        } else {
            resultsContainer.classList.remove('manual-selection-mode');
        }
        
        // Attach event listeners for song cards (handles both normal and selection modes)
        // For load more, only attach listeners to new cards starting from startIndex
        const listenerStartIndex = isLoadMore ? startIndex : 0;
        this.attachSongCardEventListeners(listenerStartIndex);
        
        // Update load more button visibility
        this.updateLoadMoreButton();
        
        // Update song count hint if export section is visible
        const exportSection = document.getElementById('export-section');
        if (exportSection && exportSection.style.display !== 'none') {
            this.updateSongCountHint();
        }
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
        const { rank, similarity, isQuery = false, fieldValue = null, embedType = null, isSelected = false } = options;
        
        let metadataHTML = '';
        if (rank && similarity !== undefined) {
            metadataHTML = `
                <div class="card-metadata">
                    <span class="card-rank">#${rank}</span>
                    <span class="similarity-score">${(similarity * 100).toFixed(1)}%</span>
                </div>
            `;
        }
        
        let tagsHTML = '';
        let playButtonHTML = '';
        if (!isQuery && song.spotify_id) {
            playButtonHTML = `<button class="song-play-btn" title="Play song">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
            </button>`;
        }
        
        if (song.tags && song.tags.length > 0) {
            const displayTags = song.tags.slice(0, 3);
            tagsHTML = `
                <div class="card-tags">
                    <div class="tags-container">
                        ${displayTags.map(tag => `<span class="tag-item">${escapeHtml(tag)}</span>`).join('')}
                    </div>
                    ${playButtonHTML}
                </div>
            `;
        } else if (playButtonHTML) {
            // If no tags but we have a play button, still create the tags container
            tagsHTML = `
                <div class="card-tags">
                    <div class="tags-container"></div>
                    ${playButtonHTML}
                </div>
            `;
        }
        
        // Accordion content based on debug mode and embedding type
        let accordionHTML = '';
        if (!isQuery) {
            let accordionContent, accordionTitle;
            
            // In debug mode, show the field value for the search embedding type
            // In production mode, always show tags + genres
            const appContainer = document.querySelector('.app-container');
            const debugMode = appContainer && appContainer.dataset.debugMode === 'true';
            if (debugMode && fieldValue !== null && fieldValue !== undefined && embedType) {
                accordionContent = this.formatFieldValueForDisplay(fieldValue, embedType);
                accordionTitle = this.getAccordionTitle(embedType);
            } else {
                // Production mode - always show tags + genres
                const tagsGenresObj = this.formatTagsGenresFromSong(song);
                accordionContent = this.formatTagsGenresForDisplay(tagsGenresObj);
                accordionTitle = this.getAccordionTitle('tags_genres');
            }
            
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
        
        // Always include checkbox, CSS will control visibility
        let checkboxHTML = '';
        if (!isQuery) {
            checkboxHTML = `<input type="checkbox" class="song-card-checkbox" ${isSelected ? 'checked' : ''}>`;
        }
        
        let footerHTML = '';
        if (rank && similarity !== undefined) {
            footerHTML = `
                <div class="card-footer">
                    <span class="card-rank">#${rank}</span>
                    <span class="similarity-score">${(similarity * 100).toFixed(1)}%</span>
                </div>
            `;
        }

        return `
            ${checkboxHTML}
            <div class="card-header">
                <img src="${escapeHtml(song.cover_url || '')}" alt="Cover" class="card-cover">
                <div class="card-info">
                    <div class="card-title">${escapeHtml(song.song)}</div>
                    <div class="card-artist">${escapeHtml(song.artist)}</div>
                    <div class="card-album">${escapeHtml(song.album || 'Unknown Album')}</div>
                </div>
            </div>
            ${tagsHTML}
            ${accordionHTML}
            ${footerHTML}
        `;
    }
    

    
    formatTagsGenresFromSong(song) {
        /**
         * Extract and clean tags and genres from song object
         * Returns an object with separate tags and genres arrays
         */
        if (!song) {
            return { tags: [], genres: [] };
        }
        
        const rawTags = song.tags || [];
        const rawGenres = song.genres || [];
        
        // Clean and filter tags
        const cleanTags = rawTags
            .filter(item => item != null) // Remove null/undefined
            .map(item => String(item).trim()) // Convert to string and trim
            .filter(item => item.length > 0); // Remove empty strings
        
        // Clean and filter genres
        const cleanGenres = rawGenres
            .filter(item => item != null) // Remove null/undefined
            .map(item => String(item).trim()) // Convert to string and trim
            .filter(item => item.length > 0); // Remove empty strings
        
        // Remove duplicates within each category
        const uniqueTags = [...new Set(cleanTags)];
        const uniqueGenres = [...new Set(cleanGenres)];
        
        return { tags: uniqueTags, genres: uniqueGenres };
    }
    
    formatTagsGenresForDisplay(tagsGenresObj) {
        /**
         * Format tags and genres with different styling
         * Takes an object with tags and genres arrays
         */
        if (!tagsGenresObj || (!tagsGenresObj.tags.length && !tagsGenresObj.genres.length)) {
            return '<div class="tags-genres-content"><em>No tags or genres available</em></div>';
        }
        
        const tagElements = tagsGenresObj.tags.map(tag => 
            `<span class="tag-item">${escapeHtml(tag)}</span>`
        ).join('');
        
        const genreElements = tagsGenresObj.genres.map(genre => 
            `<span class="genre-item">${escapeHtml(genre)}</span>`
        ).join('');
        
        return `
            <div class="tags-genres-content">
                ${tagElements}${genreElements}
            </div>
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
        document.getElementById('export-section').style.display = 'none';
        
        // Reset export accordion state
        const accordion = document.querySelector('.export-accordion');
        if (accordion) {
            accordion.classList.remove('expanded');
        }
        this.hideExportStatus();
        
        this.currentQuerySong = null;
        this.searchResults = [];
        this.originalSearchResults = [];
        this.currentSearchData = null;
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
        this.isFiltered = false;
        
        // Reset manual selection
        this.isManualSelectionMode = false;
        this.selectedSongs.clear();
        const manualSelectionToggle = document.getElementById('manual-selection-toggle');
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        // Update export form to show number input instead of selection info
        this.updateExportFormDisplay();
        
        // Reset event listener tracking
        this.resetEventListenerTracking();
        
        // Reset auto-play queue
        this.resetAutoPlayQueue();
    }
    
    resetEventListenerTracking() {
        // This method doesn't need to do anything since we clear the DOM
        // but it's here for clarity and potential future use
        console.log('🔄 Event listener tracking reset');
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
                // Auto-prompt for login if not authenticated
                this.promptForLogin();
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            this.updateAuthStatus(false);
            // Auto-prompt for login on error (likely not authenticated)
            this.promptForLogin();
        }
    }
    
    promptForLogin() {
        // Only auto-prompt once per session to avoid loops
        if (sessionStorage.getItem('autoLoginPrompted') === 'true') {
            return;
        }
        
        // Mark that we've prompted to avoid repeated prompts
        sessionStorage.setItem('autoLoginPrompted', 'true');
        
        // Show a user-friendly prompt
        const shouldLogin = confirm(
            "Welcome to Semantic Song Search! 🎵\n\n" +
            "To play songs and create playlists, you'll need to connect your Spotify account.\n\n" +
            "Would you like to login to Spotify now?"
        );
        
        if (shouldLogin) {
            // Redirect to login
            window.location.href = '/login';
        }
        // If they decline, they can still search but won't be able to play/create playlists
    }
    
    updateAuthStatus(isAuthenticated) {
        this.isAuthenticated = isAuthenticated;
        
        const indicator = document.getElementById('auth-indicator');
        const text = document.getElementById('auth-text');
        const loginBtn = document.getElementById('login-btn');
        const logoutBtn = document.getElementById('logout-btn');
        const topArtistsFilter = document.getElementById('top-artists-filter');
        const topArtistsText = document.getElementById('top-artists-text');
        
        if (isAuthenticated) {
            indicator.textContent = '●';
            indicator.className = 'auth-indicator connected';
            text.textContent = 'Connected to Spotify';
            loginBtn.style.display = 'none';
            logoutBtn.style.display = 'inline-block';
            
            // Enable top artists filter
            topArtistsFilter.disabled = false;
            topArtistsText.textContent = 'Only My Top Artists';
        } else {
            indicator.textContent = '○';
            indicator.className = 'auth-indicator';
            text.textContent = 'Not connected to Spotify';
            loginBtn.style.display = 'inline-block';
            logoutBtn.style.display = 'none';
            
            // Disable and reset top artists filter
            topArtistsFilter.disabled = true;
            topArtistsFilter.checked = false;
            topArtistsText.textContent = 'Only My Top Artists';
            this.topArtists = [];
            this.topArtistsLoaded = false;
        }
    }
    
    toggleManualSelection(enabled) {
        console.log(`🎯 Manual selection toggled: ${enabled ? 'ON' : 'OFF'}`);
        
        this.isManualSelectionMode = enabled;
        const resultsContainer = document.getElementById('results-container');
        
        if (enabled) {
            // Select all current songs by default
            this.selectedSongs.clear();
            this.searchResults.forEach((song, index) => {
                this.selectedSongs.add(song.song_idx);
            });
            console.log(`🎯 Selected ${this.selectedSongs.size} songs by default`);
            
            // Show checkboxes and enable selection styling with CSS class
            resultsContainer.classList.add('manual-selection-mode');
            this.updateAllCardSelections();
        } else {
            // Clear selections when disabled
            this.selectedSongs.clear();
            console.log(`🎯 Cleared all selections`);
            
            // Hide checkboxes and disable selection styling
            resultsContainer.classList.remove('manual-selection-mode');
            this.clearAllCardSelections();
        }
        
        // Update results count display to show/hide selection count
        this.updateResultsCount();
        
        // Update export form display
        this.updateExportFormDisplay();
        
        // Event listeners don't need to be re-attached - they handle both modes dynamically
    }
    
    updateAllCardSelections() {
        const resultsGrid = document.getElementById('results-grid');
        const songCards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        
        songCards.forEach((card, index) => {
            if (index < this.searchResults.length) {
                const song = this.searchResults[index];
                const isSelected = this.selectedSongs.has(song.song_idx);
                
                // Update checkbox state
                const checkbox = card.querySelector('.song-card-checkbox');
                if (checkbox) {
                    checkbox.checked = isSelected;
                }
                
                // Update card styling
                card.classList.add('selectable');
                if (isSelected) {
                    card.classList.add('selected');
                } else {
                    card.classList.remove('selected');
                }
            }
        });
    }
    
    clearAllCardSelections() {
        const resultsGrid = document.getElementById('results-grid');
        const songCards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        
        songCards.forEach(card => {
            // Clear checkbox state
            const checkbox = card.querySelector('.song-card-checkbox');
            if (checkbox) {
                checkbox.checked = false;
            }
            
            // Remove selection styling
            card.classList.remove('selectable', 'selected');
        });
    }
    
    // refreshSongCards method removed - no longer needed with CSS-based optimization
    
    attachSongCardEventListeners(startIndex = 0) {
        const resultsGrid = document.getElementById('results-grid');
        const songCards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        
        // Only attach listeners to cards starting from startIndex to avoid duplicates
        for (let index = startIndex; index < songCards.length && index < this.searchResults.length; index++) {
            const card = songCards[index];
            const song = this.searchResults[index];
            
            // Check if listeners are already attached to avoid duplicates
            if (card.dataset.listenersAttached === 'true') {
                continue;
            }
            
            // Mark card as having listeners attached
            card.dataset.listenersAttached = 'true';
            
            // Smart event listener that handles both modes dynamically
            card.addEventListener('click', (e) => {
                // Don't handle if clicking on the checkbox directly
                if (e.target.type === 'checkbox') return;
                
                // Don't handle if clicking on accordion elements
                if (e.target.closest('.card-accordion')) return;
                
                // Don't handle if clicking on the play button
                if (e.target.closest('.song-play-btn')) return;
                
                if (this.isManualSelectionMode) {
                    // Manual selection mode: single click toggles selection
                    this.toggleSongSelection(song.song_idx, index);
                } else {
                    // Normal mode: single click plays
                    this.playSong(song);
                }
            });
            
            // Handle play button clicks (always plays regardless of mode)
            const playButton = card.querySelector('.song-play-btn');
            if (playButton) {
                playButton.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent card click
                    this.playSong(song);
                });
            }
            
            // Handle checkbox clicks directly (always present but hidden via CSS)
            const checkbox = card.querySelector('.song-card-checkbox');
            if (checkbox) {
                checkbox.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleSongSelection(song.song_idx, index);
                });
            }
        }
    }
    
    updateExportFormDisplay() {
        const songCountField = document.getElementById('song-count-field');
        const manualSelectionInfo = document.getElementById('manual-selection-info');
        const selectedSongsCount = document.getElementById('selected-songs-count');
        
        if (this.isManualSelectionMode) {
            // Hide number input, show selection info
            songCountField.style.display = 'none';
            manualSelectionInfo.style.display = 'flex';
            
            // Update selected count
            const count = this.selectedSongs.size;
            selectedSongsCount.textContent = `${count} song${count === 1 ? '' : 's'} selected`;
        } else {
            // Show number input, hide selection info
            songCountField.style.display = 'flex';
            manualSelectionInfo.style.display = 'none';
        }
    }
    
    updateResultsCount() {
        const resultsCount = document.getElementById('results-count');
        if (!resultsCount) return;
        
        let resultsText = `${this.searchResults.length}`;
        
        // Add "filtered" if any filters are active
        if (this.isFiltered) {
            resultsText += ' filtered';
        }
        
        resultsText += ' results';
        
        // Add selected count if manual selection is active
        if (this.isManualSelectionMode) {
            const selectedCount = this.selectedSongs.size;
            resultsText += ` (${selectedCount} selected)`;
        }
        
        resultsCount.textContent = resultsText;
    }
    
    // attachAccordionEventListeners method removed - accordion listeners are now attached in displayResults directly
    
    toggleSongSelection(songIdx, cardIndex) {
        if (this.selectedSongs.has(songIdx)) {
            this.selectedSongs.delete(songIdx);
            console.log(`🎯 Deselected song ${songIdx}`);
        } else {
            this.selectedSongs.add(songIdx);
            console.log(`🎯 Selected song ${songIdx}`);
        }
        
        // Update the card's visual state
        this.updateSongCardSelection(cardIndex, this.selectedSongs.has(songIdx));
        
        console.log(`🎯 Total selected: ${this.selectedSongs.size} songs`);
        
        // Update results count display to reflect new selection
        this.updateResultsCount();
        
        // Update export form display to reflect new selection count
        this.updateExportFormDisplay();
    }
    
    updateSongCardSelection(cardIndex, isSelected) {
        const resultsGrid = document.getElementById('results-grid');
        const cards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        const card = cards[cardIndex];
        
        if (card) {
            const checkbox = card.querySelector('.song-card-checkbox');
            if (checkbox) {
                checkbox.checked = isSelected;
            }
            
            if (isSelected) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        }
    }
    
    applyClientSideFilter() {
        const topArtistsFilter = document.getElementById('top-artists-filter');
        const filterEnabled = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        console.log(`🔍 Applying client-side filter - enabled: ${filterEnabled}, top artists count: ${this.topArtists.length}`);
        
        if (!filterEnabled || !this.topArtistsLoaded || this.topArtists.length === 0) {
            // Show all original results
            console.log(`🔍 Showing all ${this.originalSearchResults.length} original results (filter disabled or no top artists)`);
            this.searchResults = [...this.originalSearchResults];
            this.isFiltered = false;
            
            // Manual selection state is preserved as-is when filter is disabled
            // No need to auto-select - user's manual selections remain unchanged
        } else {
            // Filter to only show songs by top artists
            const topArtistsSet = new Set(this.topArtists.map(artist => artist.toLowerCase()));
            console.log(`🔍 Filtering with top artists:`, Array.from(topArtistsSet).slice(0, 5), '...');
            
            const filteredOut = [];
            this.searchResults = this.originalSearchResults.filter(song => {
                const artistName = song.artist.toLowerCase().trim();
                const isMatch = topArtistsSet.has(artistName);
                if (!isMatch) {
                    filteredOut.push(`"${song.song}" by "${song.artist}"`);
                }
                return isMatch;
            });
            
            if (filteredOut.length > 0) {
                console.log(`🔍 Filtered out ${filteredOut.length} songs:`, filteredOut.slice(0, 3), filteredOut.length > 3 ? '...' : '');
            }
            
            this.isFiltered = true;
            console.log(`🔍 Filter results: ${this.searchResults.length} out of ${this.originalSearchResults.length} songs kept`);
        }
        
        // Update the display with filtered results
        const mockData = {
            results: this.searchResults,
            search_type: this.currentSearchData?.search_type || 'text',
            embed_type: this.currentSearchData?.embed_type || 'full_profile',
            query: this.currentSearchData?.query || '',
            pagination: {
                offset: 0,
                limit: this.searchResults.length,
                total_count: this.isFiltered ? null : this.originalSearchResults.length,
                has_more: false, // No more results since we're showing all filtered results
                is_filtered: this.isFiltered,
                returned_count: this.searchResults.length
            }
        };
        
        this.displayResults(mockData, false);
        
        // Sync manual selection state after filtering re-creates the cards
        if (this.isManualSelectionMode) {
            this.updateAllCardSelections();
            this.updateExportFormDisplay(); // Update export form with current selection count
        }
    }
    
    async loadTopArtists() {
        if (this.topArtistsLoaded) {
            return; // Already loaded
        }
        
        const topArtistsText = document.getElementById('top-artists-text');
        
        try {
            const response = await fetch('/api/top_artists');
            const data = await response.json();
            
            if (response.ok) {
                this.topArtists = data.top_artists;
                this.topArtistsLoaded = true;
                if (data.count === 0) {
                    topArtistsText.textContent = 'Only My Top Artists (0)';
                    console.log('⚠️ User has no top artists - may have new account or insufficient listening history');
                } else {
                    topArtistsText.textContent = `Only My Top Artists (${data.count})`;
                    console.log(`✅ Loaded ${data.count} top artists`);
                }
            } else {
                console.error('Failed to load top artists:', data.error);
                topArtistsText.textContent = 'Only My Top Artists';
                
                // If authentication error, disable the filter
                if (response.status === 401 || response.status === 403) {
                    const topArtistsFilter = document.getElementById('top-artists-filter');
                    topArtistsFilter.checked = false;
                    topArtistsFilter.disabled = true;
                    
                    // Special handling for scope insufficient error
                    if (data.requires_reauth) {
                        this.showError('Top artists filter requires additional permissions. Please logout and login again to enable this feature.');
                        this.updateAuthStatus(false);
                    }
                }
            }
        } catch (error) {
            console.error('Error loading top artists:', error);
            topArtistsText.textContent = 'Only My Top Artists';
        }
    }
    
    initSpotifyPlayer() {
        console.log('🎧 initSpotifyPlayer called');
        console.log('🎧 Access token available?', this.accessToken ? 'Yes' : 'No');
        
        if (!this.accessToken) {
            console.log('❌ No access token, skipping player initialization');
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
        
        console.log('✅ Spotify SDK loaded, creating player...');
        
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
                this.updateAuthStatus(false);
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
                this.playerInitPromise = null; // Clear the promise since we're done
                
                // Start backup auto-play checker (disabled for now to avoid interference)
                // this.startAutoPlayChecker();
                
                resolve(device_id);
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
                    this.playerInitPromise = null;
                    reject(new Error('Failed to connect to Spotify Web Playback SDK'));
                }
            }).catch(error => {
                console.error('❌ Error connecting to Spotify:', error);
                this.playerInitPromise = null;
                reject(error);
            });
        });
        
        return this.playerInitPromise;
    }
    
    async playSong(song, isAutoAdvance = false) {
        console.log('🎵 playSong called with:', song, 'isAutoAdvance:', isAutoAdvance);
        
        // Check if user is authenticated before trying to play
        if (!this.isAuthenticated || !this.accessToken) {
            // Don't auto-prompt for auto-advances (to avoid interrupting user)
            if (!isAutoAdvance) {
                const shouldLogin = confirm(
                    "You need to connect your Spotify account to play songs.\n\n" +
                    "Would you like to login to Spotify now?"
                );
                
                if (shouldLogin) {
                    window.location.href = '/login';
                    return;
                }
            }
            console.log('❌ Cannot play song - not authenticated');
            return;
        }
        
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
        
        if (!song.spotify_id) {
            console.log('❌ No Spotify ID for song');
            return;
        }
        
        if (!this.isPlayerReady) {
            console.log('🔄 Player not ready, initializing and waiting...');
            try {
                await this.initSpotifyPlayer();
                console.log('✅ Player initialization completed, proceeding with playback');
            } catch (error) {
                console.error('❌ Failed to initialize player:', error);
                return;
            }
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
            
            // Track song play
            if (typeof mixpanel !== 'undefined') {
                mixpanel.track('Song Played', {
                    'song_spotify_id': song.spotify_id || 'unknown',
                    'song_title': song.song || 'unknown',
                    'artist': song.artist || 'unknown',
                    'play_method': isAutoAdvance ? 'auto_advance' : 'manual_click',
                    'similarity_score': song.similarity || 0,
                    'position_in_results': this.searchResults.findIndex(r => r.song_idx === song.song_idx) + 1,
                    'is_authenticated': this.isAuthenticated
                });
            }
            
            if (!isAutoAdvance) { // Only count manual plays
                this.songsPlayed++;
            }
            
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
            
            // Track song play failures
            if (typeof mixpanel !== 'undefined') {
                mixpanel.track('Song Play Failed', {
                    'song_spotify_id': song.spotify_id || 'unknown',
                    'song_title': song.song || 'unknown',
                    'artist': song.artist || 'unknown',
                    'error_message': error.message || 'Unknown error',
                    'play_method': isAutoAdvance ? 'auto_advance' : 'manual_click'
                });
            }
            
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
    

    
    generatePlaylistName() {
        if (!this.currentSearchType || !this.currentQuery) {
            return 'Semantic Song Search';
        }
        
        if (this.currentSearchType === 'text') {
            // For text-to-song queries, use the query text
            const query = this.currentQuery.trim();
            if (query.length <= 100) {
                return query;
            } else {
                // Truncate to 97 chars + "..."
                return query.substring(0, 97) + '...';
            }
        } else if (this.currentSearchType === 'song' && this.currentQuerySong) {
            // For song-to-song queries, use "{song name} vibes"
            const songName = this.currentQuerySong.song || '';
            const suffix = ' vibes';
            const maxSongNameLength = 100 - suffix.length;
            
            if (songName.length <= maxSongNameLength) {
                return songName + suffix;
            } else {
                // Truncate song name to fit, keeping " vibes" intact
                const truncatedSongName = songName.substring(0, maxSongNameLength - 3) + '...';
                return truncatedSongName + suffix;
            }
        }
        
        // Fallback
        return 'Semantic Song Search';
    }
    
    toggleExportAccordion() {
        const accordion = document.querySelector('.export-accordion');
        const content = document.getElementById('export-accordion-content');
        const icon = document.querySelector('.export-accordion-icon');
        
        accordion.classList.toggle('expanded');
        
        if (accordion.classList.contains('expanded')) {
            console.log('🎵 Export accordion expanded');
            
            // Auto-populate playlist name based on current search
            const playlistNameInput = document.getElementById('playlist-name');
            if (playlistNameInput) {
                const autoName = this.generatePlaylistName();
                playlistNameInput.value = autoName;
                console.log('🎵 Auto-populated playlist name:', autoName);
            }
            
            // Check if user is authenticated when opening accordion
            if (!this.accessToken) {
                this.showExportStatus(
                    'Please <a href="/login" style="color: #1ed760; text-decoration: underline;">login to Spotify</a> to export playlists.',
                    'error'
                );
            } else {
                // Clear any previous status messages
                this.hideExportStatus();
            }
        } else {
            console.log('🎵 Export accordion collapsed');
            // Hide status when closing accordion
            this.hideExportStatus();
        }
    }
    
    async exportToPlaylist() {
        // Check if user is authenticated before trying to create playlist
        if (!this.isAuthenticated || !this.accessToken) {
            const shouldLogin = confirm(
                "You need to connect your Spotify account to create playlists.\n\n" +
                "Would you like to login to Spotify now?"
            );
            
            if (shouldLogin) {
                window.location.href = '/login';
                return;
            }
            
            this.showExportStatus('Please login to Spotify to create playlists.', 'error');
            return;
        }
        
        const playlistNameInput = document.getElementById('playlist-name');
        const songCountInput = document.getElementById('song-count');
        const exportBtn = document.getElementById('export-btn');
        const exportStatus = document.getElementById('export-status');
        const exportButtonText = document.querySelector('.export-button-text');
        const exportButtonLoading = document.querySelector('.export-button-loading');
        
        // Get input values
        const playlistName = playlistNameInput.value.trim();
        let songCount;
        
        // Validate inputs
        if (!playlistName) {
            this.showExportStatus('Please enter a playlist name.', 'error');
            return;
        }
        
        if (this.isManualSelectionMode) {
            // In manual selection mode, use the number of selected songs
            songCount = this.selectedSongs.size;
            console.log(`🎯 Manual selection mode: exporting ${songCount} selected songs`);
        } else {
            // In normal mode, validate the song count input
            songCount = parseInt(songCountInput.value);
            
            if (isNaN(songCount) || songCount < 1 || songCount > 100) {
                this.showExportStatus('Number of songs must be between 1 and 100.', 'error');
                return;
            }
            
            // Additional check for extremely large requests when auto-loading is involved
            if (songCount > this.searchResults.length && songCount > 50 && this.hasMoreResults) {
                const proceed = confirm(
                    `You requested ${songCount} songs but only ${this.searchResults.length} are currently loaded.\n\n` +
                    `This will automatically load more results, which may take some time.\n\n` +
                    `Continue with auto-loading?`
                );
                if (!proceed) {
                    return;
                }
            }
        }
        
        if (!this.searchResults || this.searchResults.length === 0) {
            this.showExportStatus('No search results available to export.', 'error');
            return;
        }
        
        // Additional validation for manual selection mode
        if (this.isManualSelectionMode && this.selectedSongs.size === 0) {
            this.showExportStatus('No songs selected. Please check at least one song to export.', 'error');
            return;
        }
        
        // Check authentication
        if (!this.accessToken) {
            console.log('🎵 No access token available for export');
            this.showExportStatus('Please <a href="/login" style="color: #1ed760; text-decoration: underline;">login to Spotify</a> first.', 'error');
            return;
        }
        
        console.log('🎵 Access token available, proceeding with export');
        
        // Check if we need to load more results
        if (songCount > this.searchResults.length && this.hasMoreResults) {
            console.log(`🎵 Requested ${songCount} songs but only have ${this.searchResults.length}. Auto-loading more results...`);
            
            // Show loading message
            this.showExportStatus(`Loading more results to reach ${songCount} songs...`, 'info');
            
            // Auto-load more results until we have enough
            await this.autoLoadMoreForExport(songCount);
        }
        
        // Prepare song IDs for export
        let songsToExport;
        if (this.isManualSelectionMode) {
            // In manual selection mode, export only selected songs
            const selectedSongsInResults = this.searchResults.filter(song => this.selectedSongs.has(song.song_idx));
            
            if (songCount > selectedSongsInResults.length) {
                console.log(`🎯 User requested ${songCount} songs but only ${selectedSongsInResults.length} are selected and available`);
                this.showExportStatus(
                    `You requested ${songCount} songs but only ${selectedSongsInResults.length} are selected. ` +
                    `Proceeding with ${selectedSongsInResults.length} tracks.`,
                    'info'
                );
            }
            
            songsToExport = selectedSongsInResults.slice(0, songCount);
            console.log(`🎯 Manual selection mode: exporting ${songsToExport.length} selected songs out of ${this.selectedSongs.size} total selected`);
        } else {
            // Normal mode: export first N songs  
            songsToExport = this.searchResults.slice(0, songCount);
            console.log(`🎵 Normal mode: exporting first ${songsToExport.length} songs`);
        }
        
        const spotifyIds = songsToExport
            .map(song => song.spotify_id)
            .filter(id => id && id.trim()); // Filter out empty/null IDs
        
        if (spotifyIds.length === 0) {
            this.showExportStatus('No valid Spotify tracks found in search results.', 'error');
            return;
        }
        
        if (spotifyIds.length < songCount) {
            console.warn(`🎵 Only ${spotifyIds.length} of ${songCount} songs have valid Spotify IDs`);
            if (spotifyIds.length < this.searchResults.length) {
                this.showExportStatus(
                    `Note: Only ${spotifyIds.length} of the first ${songCount} songs have valid Spotify IDs. ` +
                    `Proceeding with ${spotifyIds.length} tracks.`,
                    'info'
                );
                // Brief pause to show the info message
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
        
        // Show loading state
        exportBtn.disabled = true;
        exportButtonText.style.display = 'none';
        exportButtonLoading.style.display = 'inline';
        this.hideExportStatus();
        
        try {
            console.log(`🎵 Creating playlist "${playlistName}" with ${spotifyIds.length} songs`);
            
            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            // Prepare search context for tracking
            const searchContext = {
                search_type: this.currentSearchType,
                embed_type: this.currentEmbedType,
                is_filtered: this.isFiltered,
                is_manual_selection: this.isManualSelectionMode,
                selected_songs_count: this.selectedSongs.size
            };
            
            // Add query-specific context
            if (this.currentSearchType === 'text' && this.currentQuery) {
                searchContext.query = this.currentQuery;
                searchContext.query_length = this.currentQuery.length;
            } else if (this.currentSearchType === 'song' && this.currentQuerySong) {
                searchContext.query_song_idx = this.currentQuerySong.song_idx;
                searchContext.query_song_name = this.currentQuerySong.song || '';
                searchContext.query_artist_name = this.currentQuerySong.artist || '';
            }

            const response = await fetch('/api/create_playlist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    playlist_name: playlistName,
                    song_count: songCount,
                    song_spotify_ids: spotifyIds,
                    search_context: searchContext
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                const requestedText = data.track_count < songCount ? 
                    ` (${songCount} requested)` : '';
                const message = `
                    ✅ Playlist created successfully!<br>
                    <strong>${data.playlist_name}</strong><br>
                    ${data.track_count} tracks added${requestedText}<br>
                    <a href="${data.playlist_url}" target="_blank" style="color: #1ed760; text-decoration: underline; font-weight: bold;">
                        🎵 Open in Spotify ↗
                    </a>
                `;
                this.showExportStatus(message, 'success');
                console.log('🎵 Playlist created:', data);
                
                // Track successful playlist creation (for session counter only - main event tracked by backend)
                this.playlistsCreated++;
            } else {
                // Handle specific error cases
                if (response.status === 403 && data.error && 
                    (data.error.includes('permissions') || data.error.includes('Insufficient'))) {
                    // Clear the access token and prompt for re-authentication
                    console.log('🎵 Insufficient permissions detected, clearing auth state');
                    this.accessToken = null;
                    this.updateAuthStatus(false);
                    
                    // Re-check auth status since backend may have cleared session
                    setTimeout(() => this.checkAuthStatus(), 1000);
                    
                    this.showExportStatus(
                        '🔐 Playlist creation requires additional permissions.<br>' +
                        'Your current login doesn\'t have playlist creation access.<br>' +
                        '<strong>The session has been cleared.</strong><br>' +
                        '<a href="/login" style="color: #1ed760; text-decoration: underline; font-weight: bold;">Click here to re-authenticate</a> ' +
                        'with playlist permissions.',
                        'error'
                    );
                    return;
                } else if (response.status === 401) {
                    // Token expired or invalid
                    this.accessToken = null;
                    this.updateAuthStatus(false);
                    this.showExportStatus(
                        'Your Spotify session has expired. Please <a href="/login" style="color: #1ed760; text-decoration: underline;">login again</a>.',
                        'error'
                    );
                    return;
                }
                throw new Error(data.error || 'Failed to create playlist');
            }
            
        } catch (error) {
            console.error('🎵 Export error:', error);
            
            let errorMessage = 'Failed to create playlist. Please try again.';
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out. Please check your connection and try again.';
            } else if (error.message) {
                errorMessage = error.message;
            }
            
            this.showExportStatus(errorMessage, 'error');
        } finally {
            // Reset button state
            exportBtn.disabled = false;
            exportButtonText.style.display = 'inline';
            exportButtonLoading.style.display = 'none';
        }
    }
    
    showExportStatus(message, type) {
        const exportStatus = document.getElementById('export-status');
        exportStatus.innerHTML = message;
        exportStatus.className = `export-status ${type}`;
        exportStatus.style.display = 'block';
        
        // Auto-hide success messages after 10 seconds
        if (type === 'success') {
            setTimeout(() => {
                this.hideExportStatus();
            }, 10000);
        }
    }
    
    hideExportStatus() {
        const exportStatus = document.getElementById('export-status');
        exportStatus.style.display = 'none';
    }
    
    async autoLoadMoreForExport(targetCount) {
        // Keep loading more results until we have enough songs or no more results available
        while (this.searchResults.length < targetCount && this.hasMoreResults) {
            console.log(`🎵 Auto-loading more results: have ${this.searchResults.length}, need ${targetCount}`);
            
            try {
                await this.loadMoreResults();
                // Update the hint after loading more results
                this.updateSongCountHint();
                // Brief pause to prevent overwhelming the server
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error('🎵 Failed to auto-load more results:', error);
                break;
            }
        }
        
        if (this.searchResults.length < targetCount && !this.hasMoreResults) {
            console.log(`🎵 Reached end of results: have ${this.searchResults.length}, requested ${targetCount}`);
            this.showExportStatus(
                `Only ${this.searchResults.length} songs available in total. Proceeding with all available songs.`,
                'info'
            );
            // Brief pause to show the info message
            await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
            console.log(`🎵 Successfully loaded ${this.searchResults.length} results for export`);
        }
    }
    
    updateSongCountHint() {
        const songCountInput = document.getElementById('song-count');
        if (songCountInput) {
            let availableText;
            if (this.isFiltered) {
                // When filtering is active
                availableText = this.hasMoreResults ? 
                    `${this.searchResults.length} loaded (filtered)` :
                    `${this.searchResults.length} available (filtered)`;
            } else {
                // Normal case - just show current count, with "more available" if applicable
                availableText = this.hasMoreResults ? 
                    `${this.searchResults.length} loaded, more available` :
                    `${this.searchResults.length} available`;
            }
            
            songCountInput.title = `Currently ${availableText}`;
            
            // Update placeholder text
            const label = document.querySelector('label[for="song-count"]');
            if (label) {
                label.textContent = `Number of Songs (${availableText}):`;
            }
        }
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SemanticSearchApp();
});

// Handle Spotify Web Playback SDK ready callback
window.onSpotifyWebPlaybackSDKReady = () => {
    console.log('Spotify Web Playback SDK is ready');
    // Player will be initialized when needed in playSong() - no need to initialize here
}; 