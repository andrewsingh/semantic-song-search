// Semantic Song Search App JavaScript

class SemanticSearchApp {
    constructor() {
        // Initialize utilities and helpers
        this.api = new ApiHelper();
        this.analytics = new AnalyticsHelper();
        
        // Search state
        this.currentQuery = null;
        this.currentQuerySong = null;
        this.currentSearchType = 'text'; // Initialize to match HTML default
        this.searchResults = [];
        this.currentSearchData = null;
        this.lastSearchRequestData = null;
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
        this.isLoadingMore = false;
        this.searchResultsId = null;
        
        // Authentication state
        this.accessToken = null;
        this.isAuthenticated = false;
        
        // Top artists filter
        this.topArtists = [];
        this.topArtistsLoaded = false;
        this.isFiltered = false;
        this.originalSearchResults = []; // Store unfiltered results for client-side filtering
        
        // Manual song selection
        this.isManualSelectionMode = false;
        this.selectedSongs = new Set(); // Set of song indices that are selected
        
        // Personalization controls state
        this.currentLambdaVal = 0.5;
        this.currentFamiliarityMin = 0.0;
        this.currentFamiliarityMax = 1.0;
        this.hasPersonalizationHistory = false;
        
        // Cache frequently accessed DOM elements
        this.domElements = {
            topArtistsFilter: document.getElementById('top-artists-filter'),
            resultsGrid: document.getElementById('results-grid'),
            searchInput: document.getElementById('search-input'),
            suggestions: document.getElementById('suggestions'),
            familiarityMin: document.getElementById('familiarity-min'),
            familiarityMax: document.getElementById('familiarity-max'),
            exportSection: document.getElementById('export-section'),
            advancedRerunSearchBtn: document.getElementById('advanced-rerun-search-btn'),
            rerunSearchBtn: document.getElementById('rerun-search-btn'),
            manualSelectionToggle: document.getElementById('manual-selection-toggle'),
            loadMoreContainer: document.getElementById('load-more-container'),
            exportStatus: document.getElementById('export-status'),
            exportBtn: document.getElementById('export-btn'),
            embedType: document.getElementById('embed-type'),
            resultsContainer: document.getElementById('results-container'),
            resultsCount: document.getElementById('results-count'),
            resultsHeader: document.getElementById('results-header'),
            playBtn: document.getElementById('play-btn'),
            prevBtn: document.getElementById('prev-btn'),
            nextBtn: document.getElementById('next-btn'),
            lambdaSlider: document.getElementById('lambda-slider'),
            lambdaValue: document.getElementById('lambda-value'),
            loadMoreBtn: document.getElementById('load-more-btn'),
            genreSearchContainer: document.getElementById('genre-search-container'),
            genreSearchInput: document.getElementById('genre-search-input')
        };
        
        // Initialize Spotify Player
        this.player = new SpotifyPlayer(this);
        
        this.init();
    }
    
    init() {
        this.bindEventListeners();
        this.checkAuthStatus();
        
        // Ensure currentSearchType is synced with the initial HTML state
        this.currentSearchType = this.getSearchType();
        
        // Initialize genre search bar visibility based on initial search type
        if (this.currentSearchType === 'text') {
            this.domElements.genreSearchContainer.style.display = 'block';
        } else {
            this.domElements.genreSearchContainer.style.display = 'none';
        }
        
        // Track initial page load
        this.analytics.trackPageLoad();
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
    
    // Helper function to get common tracking properties
    
    bindEventListeners() {
        this.bindSearchEventListeners();
        this.bindAuthEventListeners();
        this.bindPlayerEventListeners();
        this.bindExportEventListeners();
        this.bindFilterEventListeners();
        this.bindPersonalizationEventListeners();
        this.bindAdvancedSettingsEventListeners();
        this.bindSessionEventListeners();
    }
    
    bindSearchEventListeners() {
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
        this.domElements.embedType.addEventListener('change', (e) => {
            this.handleEmbedTypeChange(e.target.value);
        });
        
        // Search input
        const searchInput = this.domElements.searchInput;
        searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });
        
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSearch();
            }
        });
        
        // Genre search input (for dual similarity)
        const genreSearchInput = this.domElements.genreSearchInput;
        genreSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                // Focus on main search input when Enter is pressed in genre search
                this.domElements.searchInput.focus();
            }
        });
        
        // Load more button
        this.domElements.loadMoreBtn.addEventListener('click', () => {
            this.loadMoreResults();
        });
    }
    
    bindAuthEventListeners() {
        // Login button
        document.getElementById('login-btn').addEventListener('click', () => {
            window.location.href = '/login';
        });
        
        // Logout button
        document.getElementById('logout-btn').addEventListener('click', () => {
            window.location.href = '/logout';
        });
    }
    
    bindPlayerEventListeners() {
        // Player controls
        this.domElements.playBtn.addEventListener('click', () => {
            this.player.togglePlayback();
        });
        
        this.domElements.prevBtn.addEventListener('click', () => {
            this.player.previousTrack();
        });
        
        this.domElements.nextBtn.addEventListener('click', () => {
            this.player.nextTrack();
        });
        
        // Progress bar
        document.getElementById('progress-bar').addEventListener('click', (e) => {
            this.player.seekToPosition(e);
        });
    }
    
    bindExportEventListeners() {
        // Export accordion toggle
        document.getElementById('export-accordion-btn').addEventListener('click', () => {
            this.toggleExportAccordion();
        });
        
        // Export button
        this.domElements.exportBtn.addEventListener('click', () => {
            this.exportToPlaylist();
        });
        
        // Manual selection toggle
        this.domElements.manualSelectionToggle.addEventListener('change', (e) => {
            this.toggleManualSelection(e.target.checked);
        });
    }
    
    bindFilterEventListeners() {
        // Top artists filter checkbox
        this.domElements.topArtistsFilter.addEventListener('change', async (e) => {
            // Track filter toggle
            this.analytics.trackEvent('Top Artists Filter Toggled', {
                'filter_enabled': e.target.checked
            });
            
            // If checked and we don't have top artists yet, load them
            if (e.target.checked && !this.topArtistsLoaded && this.isAuthenticated) {
                await this.loadTopArtists();
            }
            
            // If we have existing search results, filter them client-side instead of re-running the search
            if (this.searchResults.length > 0) {
                this.applyClientSideFilter();
            }
        });
    }
    
    bindPersonalizationEventListeners() {
        // Personalization controls
        const lambdaSlider = this.domElements.lambdaSlider;
        const familiarityMinSlider = this.domElements.familiarityMin;
        const familiarityMaxSlider = this.domElements.familiarityMax;
        const rerunSearchBtn = this.domElements.rerunSearchBtn;
        
        if (lambdaSlider) {
            lambdaSlider.addEventListener('input', (e) => {
                this.handleLambdaChange(parseFloat(e.target.value));
            });
        }
        
        if (familiarityMinSlider) {
            familiarityMinSlider.addEventListener('input', (e) => {
                this.handleFamiliarityMinChange(parseFloat(e.target.value));
            });
        }
        
        if (familiarityMaxSlider) {
            familiarityMaxSlider.addEventListener('input', (e) => {
                this.handleFamiliarityMaxChange(parseFloat(e.target.value));
            });
        }
        
        if (rerunSearchBtn) {
            rerunSearchBtn.addEventListener('click', () => {
                this.rerunSearchWithNewParameters();
            });
        }
    }
    
    bindAdvancedSettingsEventListeners() {
        // Advanced Settings Accordion
        const advancedSettingsBtn = document.getElementById('advanced-settings-accordion-btn');
        if (advancedSettingsBtn) {
            advancedSettingsBtn.addEventListener('click', () => {
                this.toggleAdvancedSettingsAccordion();
            });
        }

        // Advanced Settings Parameter Changes
        this.initAdvancedSettingsListeners();

        // Advanced Settings Rerun Search Button
        const advancedRerunBtn = this.domElements.advancedRerunSearchBtn;
        if (advancedRerunBtn) {
            advancedRerunBtn.addEventListener('click', () => {
                this.rerunSearchWithAdvancedParameters();
            });
        }

        // Reset Defaults Button
        const resetDefaultsBtn = document.getElementById('reset-defaults-btn');
        if (resetDefaultsBtn) {
            resetDefaultsBtn.addEventListener('click', async () => {
                await this.resetAdvancedParametersToDefaults();
            });
        }
    }
    
    bindSessionEventListeners() {
        // Track when user leaves the page
        window.addEventListener('beforeunload', () => {
            const sessionDuration = Math.round((Date.now() - this.analytics.sessionStartTime) / 1000);
            this.analytics.trackEvent('Session Ended', {
                'session_duration_seconds': sessionDuration,
                'searches_performed': this.analytics.searchCount || 0,
                'songs_played': this.analytics.songsPlayed || 0,
                'playlists_created': this.analytics.playlistsCreated || 0
            });
        });
    }
    
    handleSearchTypeChange(searchType) {
        const suggestionsContainer = this.domElements.suggestions;
        const querySection = document.getElementById('query-section');
        const searchInput = this.domElements.searchInput;
        
        // Track search type change
        this.analytics.trackEvent('Search Type Changed', {
            'new_search_type': searchType,
            'previous_search_type': this.currentSearchType || 'unknown'
        });
        
        // Update current search type immediately
        this.currentSearchType = searchType;
        
        if (searchType === 'song') {
            searchInput.placeholder = "🔍 Search for a song or artist... (e.g., \"Espresso\", \"Sabrina Carpenter\")";
            // Hide genre search for song-to-song search
            this.domElements.genreSearchContainer.style.display = 'none';
            this.clearResults();
        } else {
            searchInput.placeholder = "🔍 Describe the vibe you're looking for... (e.g., \"upbeat summery pop\", \"motivational workout hip hop\")";
            suggestionsContainer.style.display = 'none';
            querySection.style.display = 'none';
            // Show genre search for text-to-song search
            this.domElements.genreSearchContainer.style.display = 'block';
            this.clearResults();
        }
    }
    
    handleEmbedTypeChange(embedType) {
        
        // Track embed type change
        this.analytics.trackEvent('Embed Type Changed', {
            'new_embed_type': embedType,
            'previous_embed_type': this.currentEmbedType || 'unknown',
            'has_active_search': this.searchResults.length > 0
        });
        
        this.currentEmbedType = embedType;
        
        // Don't auto-rerun if we're currently loading more results
        if (this.isLoadingMore) {
            return;
        }
        
        // Check if we have existing search results and a query to re-run
        const query = this.domElements.searchInput.value.trim();
        const hasResults = this.searchResults.length > 0;
        const hasQuery = query.length > 0;
        
        // For song-to-song searches, also check if we have a selected query song
        const searchType = this.getSearchType();
        const hasValidQuery = hasQuery || (searchType === 'song' && this.currentQuerySong);
        
        if (hasResults && hasValidQuery) {
            // Auto-rerun the search with the new embedding type
            this.handleSearch();
        } else {
        }
    }
    
    async handleSearchInput(query) {
        const searchType = this.getSearchType();
        const suggestionsContainer = this.domElements.suggestions;
        
        if (searchType === 'song' && query.trim().length > 2) {
            try {
                const suggestions = await this.api.get(`/api/search_suggestions?query=${encodeURIComponent(query)}`);
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
        const container = this.domElements.suggestions;
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
        this.domElements.searchInput.value = suggestion.label;
        this.domElements.suggestions.style.display = 'none';
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
        const searchParams = this.extractSearchParameters();
        if (!searchParams.query) return;
        
        this.trackSearchInitiation(searchParams);
        this.searchCount++;
        
        // Handle top artists filter loading
        const filterTopArtists = await this.ensureTopArtistsLoaded(searchParams.filterTopArtists);
        
        const isNewSearch = this.prepareForNewSearch(searchParams);
        
        this.showLoading(true);
        this.hideWelcomeMessage();
        
        try {
            const requestData = this.buildSearchRequest(searchParams);
            const data = await this.performSearch(requestData);
            this.processSearchResults(data);
            
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Search failed. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    
    extractSearchParameters() {
        const searchType = this.getSearchType();
        const embedType = this.domElements.embedType.value;
        const query = this.domElements.searchInput.value.trim();
        const genreQuery = this.domElements.genreSearchInput.value.trim();
        const topArtistsFilter = this.domElements.topArtistsFilter;
        const filterTopArtists = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        return { searchType, embedType, query, genreQuery, filterTopArtists };
    }
    
    trackSearchInitiation(searchParams) {
        const { searchType, embedType, query, filterTopArtists } = searchParams;
        
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
        
        this.analytics.trackEvent('Search Initiated', searchProperties);
    }
    
    async ensureTopArtistsLoaded(filterTopArtists) {
        if (filterTopArtists && !this.topArtistsLoaded) {
            await this.loadTopArtists();
            // If loading failed, uncheck the filter and continue without it
            if (!this.topArtistsLoaded) {
                this.domElements.topArtistsFilter.checked = false;
                return false;
            }
        }
        return filterTopArtists;
    }
    
    prepareForNewSearch(searchParams) {
        const { searchType, embedType, query } = searchParams;
        
        // Create a search identifier based on actual search parameters
        const newSearchId = `${searchType}:${embedType}:${query}:${this.currentQuerySong?.song_idx || ''}`;
        const isNewSearch = this.searchResultsId !== newSearchId;
        
        // Reset pagination for new searches
        this.currentOffset = 0;
        this.searchResults = [];
        this.originalSearchResults = [];
        
        // Reset manual selection for new searches
        this.isManualSelectionMode = false;
        this.selectedSongs.clear();
        const manualSelectionToggle = this.domElements.manualSelectionToggle;
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        this.updateExportFormDisplay();
        
        // Reset queue when starting a new search
        if (isNewSearch) {
            this.searchResultsId = newSearchId;
            this.player.resetAutoPlayQueue();
        }
        
        // Store current search parameters for playlist name auto-population
        this.currentQuery = query;
        this.currentSearchType = searchType;
        
        return isNewSearch;
    }
    
    buildSearchRequest(searchParams) {
        const { searchType, embedType, query, genreQuery } = searchParams;
        
        // Get advanced parameters
        const advancedParams = this.getActiveAdvancedParams();
        
        const requestData = {
            search_type: searchType,
            embed_type: embedType,
            query: query,
            k: 20,
            offset: 0,
            // Personalization parameters  
            lambda_val: this.activeLambdaVal !== undefined ? this.activeLambdaVal : (this.currentLambdaVal !== undefined ? this.currentLambdaVal : 0.5),
            familiarity_min: this.activeFamiliarityMin !== undefined ? this.activeFamiliarityMin : (this.currentFamiliarityMin !== undefined ? this.currentFamiliarityMin : 0.0),
            familiarity_max: this.activeFamiliarityMax !== undefined ? this.activeFamiliarityMax : (this.currentFamiliarityMax !== undefined ? this.currentFamiliarityMax : 1.0),
            // Advanced ranking parameters
            ...advancedParams
        };
        
        // Add genre query if provided
        if (genreQuery) {
            requestData.genre_query = genreQuery;
        }
        
        if (searchType === 'song' && this.currentQuerySong) {
            requestData.song_idx = this.currentQuerySong.song_idx;
        }
        
        // Store current search data for load more functionality and weight updates
        this.currentSearchData = { ...requestData };
        this.lastSearchRequestData = { ...requestData };
        
        return requestData;
    }
    
    async performSearch(requestData) {
        const data = await this.api.post('/api/search', requestData);
        
        // Validate API response structure
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid API response format');
        }
        
        if (!Array.isArray(data.results)) {
            throw new Error('API response missing results array');
        }
        
        return data;
    }
    
    processSearchResults(data) {
        this.searchResults = data.results;
        this.originalSearchResults = [...data.results]; // Store original unfiltered results for client-side filtering
        this.currentOffset = data.pagination.offset + data.pagination.limit;
        this.totalResultsCount = data.pagination.total_count;
        this.hasMoreResults = data.pagination.has_more;
        this.isFiltered = false; // Reset since server doesn't filter anymore
        
        // Apply client-side filter if checkbox is currently checked
        const topArtistsFilter = this.domElements.topArtistsFilter;
        if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.topArtistsLoaded) {
            this.applyClientSideFilter();
        } else {
            this.displayResults(data, false);
        }
    }
    
    async loadMoreResults() {
        if (this.isLoadingMore || !this.hasMoreResults || !this.currentSearchData) {
            return;
        }
        
        // Track load more click
        this.analytics.trackEvent('Load More Clicked', {
            'current_results_count': this.searchResults.length,
            'current_offset': this.currentOffset,
            'search_type': this.currentSearchData.search_type,
            'embed_type': this.currentSearchData.embed_type,
            'has_filter_active': this.domElements.topArtistsFilter.checked,
            'is_manual_selection_mode': this.isManualSelectionMode
        });
        
        this.isLoadingMore = true;
        this.showLoadMoreLoading(true);
        
        try {
            const requestData = {
                ...this.currentSearchData,
                offset: this.currentOffset,
                // Use current personalization parameters (may have changed since original search)
                lambda_val: this.currentLambdaVal !== undefined ? this.currentLambdaVal : 0.5,
                familiarity_min: this.currentFamiliarityMin !== undefined ? this.currentFamiliarityMin : 0.0,
                familiarity_max: this.currentFamiliarityMax !== undefined ? this.currentFamiliarityMax : 1.0
            };
            
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                let errorMessage = `Load more failed: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    console.error('🔄 Load more error response:', errorData);
                    if (errorData.error) {
                        errorMessage = `Load more failed: ${errorData.error}`;
                    }
                } catch (e) {
                    console.error('🔄 Failed to parse error response');
                }
                throw new Error(errorMessage);
            }
            
            const data = await response.json();
            
            // Validate load more response
            if (!data || !Array.isArray(data.results)) {
                throw new Error('Invalid load more response format');
            }
            
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
            }
            
            // Check if we have client-side filtering active
            const topArtistsFilter = this.domElements.topArtistsFilter;
            if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.topArtistsLoaded) {
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
            
            console.log('🎵 Added', data.results.length, 'new songs to queue (was', previousResultsCount, ')');
            
            // If user was at the end of queue and new results loaded, they can now continue
            if (this.currentSongIndex === previousResultsCount - 1 && data.results.length > 0) {
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
        const resultsHeader = this.domElements.resultsHeader;
        const resultsCount = this.domElements.resultsCount;
        const searchInfo = document.getElementById('search-info');
        const resultsGrid = this.domElements.resultsGrid;
        const loadMoreContainer = this.domElements.loadMoreContainer;
        
        // Update header (show current results count with new formatting)
        this.updateResultsCount();
        
        // Create enhanced search info with ranking details
        let searchInfoText = `${data.search_type} search • ${data.embed_type.replace('_', ' ')}`;
        
        // Add ranking weights information if available
        if (data.ranking_weights) {
            const weights = data.ranking_weights;
            if (weights.has_history) {
                searchInfoText += ` • Personalized ranking (${weights.history_songs_count} songs)`;
            } else {
                searchInfoText += ` • Semantic similarity only`;
            }
        }
        
        searchInfo.textContent = searchInfoText;
        resultsHeader.style.display = 'flex';
        
        // Display ranking weights if available (only for new searches)
        if (!isLoadMore) {
            this.displayRankingWeights(data.ranking_weights);
            
            // Show/hide personalization controls based on history availability
            const hasHistory = data.ranking_weights && data.ranking_weights.has_history;
            this.showPersonalizationControls(hasHistory);
        }
        
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
            this.domElements.exportSection.style.display = 'none';
            return;
        }
        
        // Show export section when results are available (only for new searches)
        if (!isLoadMore && this.searchResults.length > 0) {
            this.domElements.exportSection.style.display = 'block';
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
        const resultsContainer = this.domElements.resultsContainer;
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
        const exportSection = this.domElements.exportSection;
        if (exportSection && exportSection.style.display !== 'none') {
            this.updateSongCountHint();
        }
    }
    
    updateLoadMoreButton() {
        const loadMoreContainer = this.domElements.loadMoreContainer;
        const loadMoreBtn = this.domElements.loadMoreBtn;
        
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
    
    displayRankingWeights(rankingWeights) {
        let rankingWeightsContainer = document.getElementById('ranking-weights-container');
        if (rankingWeightsContainer) {
            rankingWeightsContainer.style.display = 'none';
        }
    }
    
    toggleWeightsEditing() {
        const editBtn = document.getElementById('edit-weights-btn');
        const weightValues = document.querySelectorAll('.weight-value[data-weight]');
        
        if (editBtn.textContent === 'Edit') {
            // Enter edit mode
            editBtn.textContent = 'Save & Run';
            editBtn.classList.add('editing');
            
            // Convert weight values to input fields
            weightValues.forEach(valueSpan => {
                const currentValue = valueSpan.textContent.replace('%', '');
                const weightKey = valueSpan.dataset.weight;
                
                const input = document.createElement('input');
                input.type = 'number';
                input.step = '0.1';
                input.min = '0';
                input.max = '100';
                input.value = currentValue;
                input.className = 'weight-input';
                input.dataset.weight = weightKey;
                
                valueSpan.replaceWith(input);
            });
        } else {
            // Save and rerank
            this.saveWeightsAndRerank();
        }
    }
    
    async saveWeightsAndRerank() {
        const editBtn = document.getElementById('edit-weights-btn');
        const weightInputs = document.querySelectorAll('.weight-input[data-weight]');
        
        // Collect new weights
        const newWeights = {};
        let totalWeight = 0;
        
        weightInputs.forEach(input => {
            const weight = (parseFloat(input.value) || 0) / 100; // Convert percentage to decimal
            newWeights[input.dataset.weight] = weight;
            totalWeight += weight;
        });
        
        // Validate weights sum to approximately 100%
        const totalPercentage = totalWeight * 100;
        if (Math.abs(totalPercentage - 100.0) > 10) {
            alert('Warning: Weights should sum to approximately 100%. Current sum: ' + totalPercentage.toFixed(1) + '%');
            return;
        }
        
        try {
            // Update button state
            editBtn.textContent = 'Updating...';
            editBtn.disabled = true;
            
            // Update weights on server
            await this.updateServerWeights(newWeights);
            
            // Trigger new search with updated weights
            await this.performNewSearchWithUpdatedWeights();
            
            // Exit edit mode
            editBtn.textContent = 'Edit';
            editBtn.classList.remove('editing');
            editBtn.disabled = false;
            
            // Convert inputs back to spans
            weightInputs.forEach(input => {
                const valueSpan = document.createElement('span');
                valueSpan.className = 'weight-value';
                valueSpan.dataset.weight = input.dataset.weight;
                valueSpan.textContent = parseFloat(input.value).toFixed(1) + '%';
                
                input.replaceWith(valueSpan);
            });
            
        } catch (error) {
            console.error('Error updating weights:', error);
            alert('Error updating weights: ' + error.message);
            editBtn.textContent = 'Save & Run';
            editBtn.disabled = false;
        }
    }
    
    async updateServerWeights(newWeights) {
        return await this.api.put('/api/ranking_weights', newWeights);
    }
    
    async performNewSearchWithUpdatedWeights() {
        // Check if we have a current search to repeat
        if (!this.lastSearchRequestData) {
            throw new Error('No current search to repeat with new weights');
        }
        
        // Repeat the same search with updated server weights
        const requestData = this.lastSearchRequestData;
        
        // Clear current results and reset state
        this.searchResults = [];
        this.originalSearchResults = [];
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
        
        // Perform the search API call directly
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
        
        // Process the results same as original search
        this.searchResults = data.results;
        this.originalSearchResults = [...data.results];
        this.currentOffset = data.pagination.offset + data.pagination.limit;
        this.totalResultsCount = data.pagination.total_count;
        this.hasMoreResults = data.pagination.has_more;
        this.isFiltered = false;
        // Note: Keep original currentSearchData with request parameters for load more
        
        // Re-render the results
        this.displayResults(data, false); // isLoadMore = false
        this.displayRankingWeights(data.ranking_weights);
        this.updateResultsCount();
        
        // Re-apply any active filters
        const topArtistsFilter = this.domElements.topArtistsFilter;
        if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.topArtistsLoaded) {
            this.applyClientSideFilter();
        }
    }
    
    async rerankResultsWithNewWeights(newWeights) {
        // Apply new weights to existing search results
        this.searchResults.forEach(song => {
            if (song.scoring_components) {
                const components = song.scoring_components;
                
                // Recalculate weighted components
                components.semantic_weighted = newWeights.w_sem * components.semantic_similarity;
                components.interest_weighted = newWeights.w_int * components.personal_interest;
                components.exploration_weighted = newWeights.w_ucb * components.exploration_bonus;
                components.popularity_weighted = newWeights.w_pop * components.popularity_score;
                
                // Recalculate final score
                song.final_score = components.semantic_weighted + components.interest_weighted + 
                                 components.exploration_weighted + components.popularity_weighted;
            }
        });
        
        // Also update originalSearchResults
        this.originalSearchResults.forEach(song => {
            if (song.scoring_components) {
                const components = song.scoring_components;
                
                // Recalculate weighted components
                components.semantic_weighted = newWeights.w_sem * components.semantic_similarity;
                components.interest_weighted = newWeights.w_int * components.personal_interest;
                components.exploration_weighted = newWeights.w_ucb * components.exploration_bonus;
                components.popularity_weighted = newWeights.w_pop * components.popularity_score;
                
                // Recalculate final score
                song.final_score = components.semantic_weighted + components.interest_weighted + 
                                 components.exploration_weighted + components.popularity_weighted;
            }
        });
        
        // Re-sort results by new final scores
        this.searchResults.sort((a, b) => (b.final_score || 0) - (a.final_score || 0));
        this.originalSearchResults.sort((a, b) => (b.final_score || 0) - (a.final_score || 0));
        
        // Re-apply current filter if active
        this.applyClientSideFilter();
        
        // Update the UI with re-ranked results
        const mockData = {
            results: this.searchResults,
            search_type: this.currentSearchData?.search_type || 'text',
            embed_type: this.currentSearchData?.embed_type || 'tags_genres',
            query: this.currentSearchData?.query || '',
            ranking_weights: {
                ...newWeights,
                has_history: this.currentSearchData?.ranking_weights?.has_history || false,
                history_songs_count: this.currentSearchData?.ranking_weights?.history_songs_count || 0
            },
            pagination: {
                offset: 0,
                limit: this.searchResults.length,
                total_count: this.searchResults.length,
                has_more: false,
                returned_count: this.searchResults.length
            }
        };
        
        // Update current search data
        this.currentSearchData = mockData;
        
        // Re-render results
        this.displayResults(mockData, false);
    }
    
    createSongCardHTML(song, options = {}) {
        const { rank, similarity, isQuery = false, fieldValue = null, embedType = null, isSelected = false } = options;
        
        
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
        if (rank && (similarity !== undefined || song.final_score !== undefined)) {
            const finalScore = song.final_score !== undefined ? song.final_score : similarity;
            
            // Build scoring components for the middle section (only if we have meaningful breakdown)
            let scoringComponentsHTML = '';
            if (song.scoring_components) {
                const components = song.scoring_components;
                
                scoringComponentsHTML = `
                    <div class="scoring-components">
                        <span class="scoring-component" title="Track-level similarity">
                            <span class="component-label">Sim:</span>
                            <span class="component-value">${(components.S_track || 0).toFixed(2)}</span>
                        </span>
                        <span class="scoring-component" title="Genre similarity">
                            <span class="component-label">Gen:</span>
                            <span class="component-value">${(components.S_genre || 0).toFixed(2)}</span>
                        </span>
                        <span class="scoring-component" title="Artist popularity vibe similarity">
                            <span class="component-label">Art:</span>
                            <span class="component-value">${(components.S_artist_pop || 0).toFixed(2)}</span>
                        </span>
                        <span class="scoring-component" title="Popularity score">
                            <span class="component-label">Pop:</span>
                            <span class="component-value">${(components.S_pop || 0).toFixed(2)}</span>
                        </span>
                    </div>
                `;
            }
            
            footerHTML = `
                <div class="card-footer">
                    <span class="card-rank">#${rank}</span>
                    ${scoringComponentsHTML}
                    <span class="similarity-score">${(finalScore * 100).toFixed(1)}</span>
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
        this.domElements.resultsHeader.style.display = 'none';
        this.domElements.resultsGrid.innerHTML = '';
        this.domElements.loadMoreContainer.style.display = 'none';
        document.getElementById('welcome-message').style.display = 'block';
        this.domElements.exportSection.style.display = 'none';
        
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
        this.lastSearchRequestData = null;
        this.currentOffset = 0;
        this.totalResultsCount = 0;
        this.hasMoreResults = false;
        this.isFiltered = false;
        
        // Reset manual selection
        this.isManualSelectionMode = false;
        this.selectedSongs.clear();
        const manualSelectionToggle = this.domElements.manualSelectionToggle;
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        // Update export form to show number input instead of selection info
        this.updateExportFormDisplay();
        
        // Reset event listener tracking
        this.resetEventListenerTracking();
        
        // Reset auto-play queue
        this.player.resetAutoPlayQueue();
    }
    
    resetEventListenerTracking() {
        // This method doesn't need to do anything since we clear the DOM
        // but it's here for clarity and potential future use
    }
    
    
    // Spotify Authentication and Player
    async checkAuthStatus() {
        try {
            const data = await this.api.get('/api/token');
            this.accessToken = data.access_token;
            this.updateAuthStatus(true);
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
        const topArtistsFilter = this.domElements.topArtistsFilter;
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
        
        this.isManualSelectionMode = enabled;
        const resultsContainer = this.domElements.resultsContainer;
        
        if (enabled) {
            // Select all current songs by default
            this.selectedSongs.clear();
            this.searchResults.forEach((song, index) => {
                this.selectedSongs.add(song.song_idx);
            });
            
            // Show checkboxes and enable selection styling with CSS class
            resultsContainer.classList.add('manual-selection-mode');
            this.updateAllCardSelections();
        } else {
            // Clear selections when disabled
            this.selectedSongs.clear();
            
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
        const resultsGrid = this.domElements.resultsGrid;
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
        const resultsGrid = this.domElements.resultsGrid;
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
    
    
    attachSongCardEventListeners(startIndex = 0) {
        const resultsGrid = this.domElements.resultsGrid;
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
                    this.player.playSong(song);
                }
            });
            
            // Handle play button clicks (always plays regardless of mode)
            const playButton = card.querySelector('.song-play-btn');
            if (playButton) {
                playButton.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent card click
                    this.player.playSong(song);
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
        const resultsCount = this.domElements.resultsCount;
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
    
    
    toggleSongSelection(songIdx, cardIndex) {
        if (this.selectedSongs.has(songIdx)) {
            this.selectedSongs.delete(songIdx);
        } else {
            this.selectedSongs.add(songIdx);
        }
        
        // Update the card's visual state
        this.updateSongCardSelection(cardIndex, this.selectedSongs.has(songIdx));
        
        
        // Update results count display to reflect new selection
        this.updateResultsCount();
        
        // Update export form display to reflect new selection count
        this.updateExportFormDisplay();
    }
    
    updateSongCardSelection(cardIndex, isSelected) {
        const resultsGrid = this.domElements.resultsGrid;
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
        const topArtistsFilter = this.domElements.topArtistsFilter;
        const filterEnabled = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        
        if (!filterEnabled || !this.topArtistsLoaded || !this.topArtists || this.topArtists.length === 0) {
            // Show all original results
            console.log(`🔍 Showing all ${this.originalSearchResults.length} original results (filter disabled or no top artists)`);
            this.searchResults = this.originalSearchResults; // Reference, not copy
            this.isFiltered = false;
            
            // Manual selection state is preserved as-is when filter is disabled
            // No need to auto-select - user's manual selections remain unchanged
        } else {
            // Filter to only show songs by top artists
            const topArtistsSet = new Set(this.topArtists.map(artist => artist.name.toLowerCase()));
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
        }
        
        // If in manual selection mode, remove filtered-out songs from selection
        if (this.isManualSelectionMode && this.selectedSongs.size > 0) {
            // Create a set of currently visible song indices for efficient lookup
            const visibleSongIndices = new Set(this.searchResults.map(song => song.song_idx));
            
            // Remove any selected songs that are no longer visible
            const originalSelectionSize = this.selectedSongs.size;
            for (const songIdx of this.selectedSongs) {
                if (!visibleSongIndices.has(songIdx)) {
                    this.selectedSongs.delete(songIdx);
                }
            }
            
            const removedCount = originalSelectionSize - this.selectedSongs.size;
            if (removedCount > 0) {
                console.log(`🎯 Removed ${removedCount} filtered-out songs from selection (${this.selectedSongs.size} songs still selected)`);
            }
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
            this.updateResultsCount(); // Update results count display
        }
    }
    
    async loadTopArtists() {
        if (this.topArtistsLoaded) {
            return; // Already loaded
        }
        
        const topArtistsText = document.getElementById('top-artists-text');
        
        try {
            const data = await this.api.get('/api/top_artists');
            this.topArtists = data.artists || [];
            this.topArtistsLoaded = true;
            const count = this.topArtists.length;
            if (count === 0) {
                topArtistsText.textContent = 'Only My Top Artists (0)';
            } else {
                topArtistsText.textContent = `Only My Top Artists (${count})`;
            }
        } catch (error) {
            console.error('Error loading top artists:', error);
            topArtistsText.textContent = 'Only My Top Artists';
        }
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
            
            // Auto-populate playlist name based on current search
            const playlistNameInput = document.getElementById('playlist-name');
            if (playlistNameInput) {
                const autoName = this.generatePlaylistName();
                playlistNameInput.value = autoName;
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
            // Hide status when closing accordion
            this.hideExportStatus();
        }
    }
    
    async exportToPlaylist() {
        const validation = this.validateExportInputs();
        if (!validation.isValid) {
            return;
        }
        
        const { playlistName, songCount } = validation;
        const exportElements = this.getExportElements();
        
        const spotifyIds = await this.prepareSongsForExport(songCount);
        if (!spotifyIds) {
            return;
        }
        
        await this.performPlaylistCreation(playlistName, songCount, spotifyIds, exportElements);
    }
    
    validateExportInputs() {
        // Check if user is authenticated before trying to create playlist
        if (!this.isAuthenticated || !this.accessToken) {
            const shouldLogin = confirm(
                "You need to connect your Spotify account to create playlists.\n\n" +
                "Would you like to login to Spotify now?"
            );
            
            if (shouldLogin) {
                window.location.href = '/login';
                return { isValid: false };
            }
            
            this.showExportStatus('Please login to Spotify to create playlists.', 'error');
            return { isValid: false };
        }
        
        const playlistNameInput = document.getElementById('playlist-name');
        const songCountInput = document.getElementById('song-count');
        
        // Get input values
        const playlistName = playlistNameInput.value.trim();
        let songCount;
        
        // Validate inputs
        if (!playlistName) {
            this.showExportStatus('Please enter a playlist name.', 'error');
            return { isValid: false };
        }
        
        if (this.isManualSelectionMode) {
            // In manual selection mode, use the number of selected songs
            songCount = this.selectedSongs.size;
        } else {
            // In normal mode, validate the song count input
            songCount = parseInt(songCountInput.value);
            
            if (isNaN(songCount) || songCount < 1 || songCount > 100) {
                this.showExportStatus('Number of songs must be between 1 and 100.', 'error');
                return { isValid: false };
            }
            
            // Additional check for extremely large requests when auto-loading is involved
            if (songCount > this.searchResults.length && songCount > 50 && this.hasMoreResults) {
                const proceed = confirm(
                    `You requested ${songCount} songs but only ${this.searchResults.length} are currently loaded.\n\n` +
                    `This will automatically load more results, which may take some time.\n\n` +
                    `Continue with auto-loading?`
                );
                if (!proceed) {
                    return { isValid: false };
                }
            }
        }
        
        if (!this.searchResults || this.searchResults.length === 0) {
            this.showExportStatus('No search results available to export.', 'error');
            return { isValid: false };
        }
        
        // Additional validation for manual selection mode
        if (this.isManualSelectionMode && this.selectedSongs.size === 0) {
            this.showExportStatus('No songs selected. Please check at least one song to export.', 'error');
            return { isValid: false };
        }
        
        // Check authentication
        if (!this.accessToken) {
            this.showExportStatus('Please <a href="/login" style="color: #1ed760; text-decoration: underline;">login to Spotify</a> first.', 'error');
            return { isValid: false };
        }
        
        return { isValid: true, playlistName, songCount };
    }
    
    getExportElements() {
        return {
            exportBtn: this.domElements.exportBtn,
            exportStatus: this.domElements.exportStatus,
            exportButtonText: document.querySelector('.export-button-text'),
            exportButtonLoading: document.querySelector('.export-button-loading')
        };
    }
    
    async prepareSongsForExport(songCount) {
        // Check if we need to load more results
        if (songCount > this.searchResults.length && this.hasMoreResults) {
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
                this.showExportStatus(
                    `You requested ${songCount} songs but only ${selectedSongsInResults.length} are selected. ` +
                    `Proceeding with ${selectedSongsInResults.length} tracks.`,
                    'info'
                );
            }
            
            songsToExport = selectedSongsInResults.slice(0, songCount);
        } else {
            // Normal mode: export first N songs  
            songsToExport = this.searchResults.slice(0, songCount);
        }
        
        const spotifyIds = songsToExport
            .map(song => song.spotify_id)
            .filter(id => id && id.trim()); // Filter out empty/null IDs
        
        if (spotifyIds.length === 0) {
            this.showExportStatus('No valid Spotify tracks found in search results.', 'error');
            return null;
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
        
        return spotifyIds;
    }
    
    async performPlaylistCreation(playlistName, songCount, spotifyIds, exportElements) {
        const { exportBtn, exportButtonText, exportButtonLoading } = exportElements;
        
        // Show loading state
        exportBtn.disabled = true;
        exportButtonText.style.display = 'none';
        exportButtonLoading.style.display = 'inline';
        this.hideExportStatus();
        
        try {
            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            // Prepare search context for tracking
            const searchContext = this.buildSearchContext();
            
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
                this.handlePlaylistCreationSuccess(data, songCount);
            } else {
                this.handlePlaylistCreationError(response, data);
            }
            
        } catch (error) {
            this.handlePlaylistCreationException(error);
        } finally {
            // Reset button state
            exportBtn.disabled = false;
            exportButtonText.style.display = 'inline';
            exportButtonLoading.style.display = 'none';
        }
    }
    
    buildSearchContext() {
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
        
        return searchContext;
    }
    
    handlePlaylistCreationSuccess(data, songCount) {
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
        
        // Track successful playlist creation (for session counter only - main event tracked by backend)
        this.playlistsCreated++;
    }
    
    handlePlaylistCreationError(response, data) {
        // Handle specific error cases
        if (response.status === 403 && data.error && 
            (data.error.includes('permissions') || data.error.includes('Insufficient'))) {
            // Clear the access token and prompt for re-authentication
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
    
    handlePlaylistCreationException(error) {
        console.error('🎵 Export error:', error);
        
        let errorMessage = 'Failed to create playlist. Please try again.';
        if (error.name === 'AbortError') {
            errorMessage = 'Request timed out. Please check your connection and try again.';
        } else if (error.message) {
            errorMessage = error.message;
        }
        
        this.showExportStatus(errorMessage, 'error');
    }
    
    showExportStatus(message, type) {
        const exportStatus = this.domElements.exportStatus;
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
        const exportStatus = this.domElements.exportStatus;
        exportStatus.style.display = 'none';
    }
    
    async autoLoadMoreForExport(targetCount) {
        // Keep loading more results until we have enough songs or no more results available
        while (this.searchResults.length < targetCount && this.hasMoreResults) {
            
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
            this.showExportStatus(
                `Only ${this.searchResults.length} songs available in total. Proceeding with all available songs.`,
                'info'
            );
            // Brief pause to show the info message
            await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
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
            
            const label = document.querySelector('label[for="song-count"]');
            if (label) {
                label.textContent = `Number of Songs (${availableText}):`;
            }
        }
    }
    
    // Personalization Controls Handlers
    handleLambdaChange(value) {
        this.currentLambdaVal = value;
        const lambdaDisplay = this.domElements.lambdaValue;
        if (lambdaDisplay) {
            lambdaDisplay.textContent = value.toFixed(2);
            // Position above the lambda slider knob using accurate positioning
            const slider = this.domElements.lambdaSlider;
            this.positionSliderValue(slider, lambdaDisplay, value, 0, 1);
        }
        this.enableRerunButton();
        
        // Track lambda change
        this.analytics.trackEvent('Lambda Value Changed', {
            'lambda_val': value,
            'has_active_search': this.searchResults.length > 0
        });
    }
    
    handleFamiliarityMinChange(value) {
        const maxSlider = this.domElements.familiarityMax;
        const currentMax = parseFloat(maxSlider.value);
        
        // Ensure min doesn't exceed max
        if (value > currentMax) {
            value = currentMax;
            this.domElements.familiarityMin.value = value;
        }
        
        this.currentFamiliarityMin = value;
        this.updateFamiliarityRangeDisplay();
        this.enableRerunButton();
    }
    
    handleFamiliarityMaxChange(value) {
        const minSlider = this.domElements.familiarityMin;
        const currentMin = parseFloat(minSlider.value);
        
        // Ensure max doesn't go below min
        if (value < currentMin) {
            value = currentMin;
            this.domElements.familiarityMax.value = value;
        }
        
        this.currentFamiliarityMax = value;
        this.updateFamiliarityRangeDisplay();
        this.enableRerunButton();
    }
    
    updateFamiliarityRangeDisplay() {
        const minDisplay = document.getElementById('familiarity-min-value');
        const maxDisplay = document.getElementById('familiarity-max-value');
        
        if (minDisplay) {
            minDisplay.textContent = this.currentFamiliarityMin.toFixed(2);
            // Position above the min slider knob using accurate positioning
            const minSlider = this.domElements.familiarityMin;
            this.positionSliderValue(minSlider, minDisplay, this.currentFamiliarityMin, 0, 1);
        }
        if (maxDisplay) {
            maxDisplay.textContent = this.currentFamiliarityMax.toFixed(2);
            // Position above the max slider knob using accurate positioning
            const maxSlider = this.domElements.familiarityMax;
            this.positionSliderValue(maxSlider, maxDisplay, this.currentFamiliarityMax, 0, 1);
        }
    }
    
    // Accurately position slider value display above the slider thumb
    positionSliderValue(slider, valueDisplay, value, min, max) {
        if (!slider || !valueDisplay) return;
        
        // Calculate the percentage position of the value within the range
        const percent = (value - min) / (max - min);
        
        // Get the slider's bounding rectangle
        const sliderRect = slider.getBoundingClientRect();
        
        // Estimate thumb width (typically 20px for most browsers)
        const thumbWidth = 20;
        
        // Calculate the effective track width (slider width minus thumb width)
        const trackWidth = sliderRect.width - thumbWidth;
        
        // Calculate position: start at half thumb width, then add track progress
        const position = (thumbWidth / 2) + (percent * trackWidth);
        
        // Convert to percentage of total slider width
        const positionPercent = (position / sliderRect.width) * 100;
        
        valueDisplay.style.left = `${positionPercent}%`;
    }
    
    initializeSliderPositions() {
        // Initialize lambda slider position
        const lambdaDisplay = this.domElements.lambdaValue;
        if (lambdaDisplay) {
            const lambdaPercent = (this.currentLambdaVal - 0) / (1 - 0) * 100;
            lambdaDisplay.style.left = `${lambdaPercent}%`;
        }
        
        // Initialize familiarity range positions
        this.updateFamiliarityRangeDisplay();
    }
    
    enableRerunButton() {
        const rerunBtn = this.domElements.rerunSearchBtn;
        if (rerunBtn) {
            rerunBtn.disabled = false;
        }
    }
    
    disableRerunButton() {
        const rerunBtn = this.domElements.rerunSearchBtn;
        if (rerunBtn) {
            rerunBtn.disabled = true;
        }
    }
    
    async rerunSearchWithNewParameters() {
        
        // Save current values as the "active" values
        this.activeLambdaVal = this.currentLambdaVal;
        this.activeFamiliarityMin = this.currentFamiliarityMin;
        this.activeFamiliarityMax = this.currentFamiliarityMax;
        
        // Disable the rerun button
        this.disableRerunButton();
        
        // Track parameter update
        this.analytics.trackEvent('Personalization Parameters Updated', {
            'lambda_val': this.activeLambdaVal,
            'familiarity_min': this.activeFamiliarityMin,
            'familiarity_max': this.activeFamiliarityMax,
            'has_active_search': this.searchResults.length > 0
        });
        
        // Rerun the current search if we have one
        if (this.lastSearchRequestData) {
            await this.handleSearch();
        }
    }
    
    // Advanced Settings Methods
    toggleAdvancedSettingsAccordion() {
        const accordion = document.querySelector('.advanced-settings-accordion');
        const content = document.getElementById('advanced-settings-accordion-content');
        
        if (accordion && content) {
            accordion.classList.toggle('expanded');
            
            // Initialize parameter values if opening for the first time
            if (accordion.classList.contains('expanded') && !this.advancedSettingsInitialized) {
                this.initializeAdvancedSettingsValues();
                this.advancedSettingsInitialized = true;
            }
        }
    }
    
    initAdvancedSettingsListeners() {
        // Get all parameter input elements
        const parameterIds = [
            'H_c', 'H_E', 'gamma_s', 'gamma_f', 'kappa', 'alpha_0', 'beta_0', 'K_s',
            'K_E', 'gamma_A', 'eta', 'tau', 'beta_f', 'K_life', 'K_recent', 'psi',
            'k_neighbors', 'sigma', 'knn_embed_type', 'beta_p', 'beta_s', 'beta_a',
            'kappa_E', 'theta_c', 'tau_c', 'K_c', 'tau_K', 'M_A', 'K_fam', 'R_min',
            'C_fam', 'min_plays', 'beta_track', 'beta_artist_pop', 'beta_artist_personal', 'beta_genre', 'beta_pop'
        ];
        
        // Initialize current advanced parameters object
        this.currentAdvancedParams = {};
        this.activeAdvancedParams = {};
        
        // Add event listeners for each parameter
        parameterIds.forEach(paramId => {
            const element = document.getElementById(paramId);
            if (element) {
                element.addEventListener('input', (e) => {
                    this.handleAdvancedParameterChange(paramId, e.target.value);
                });
            }
        });
    }
    
    handleAdvancedParameterChange(paramId, value) {
        // Convert value to appropriate type
        let convertedValue;
        if (paramId === 'knn_embed_type') {
            convertedValue = value;
        } else if (paramId === 'k_neighbors' || paramId === 'min_plays') {
            convertedValue = parseInt(value);
        } else {
            convertedValue = parseFloat(value);
        }
        
        // Store the current value
        this.currentAdvancedParams[paramId] = convertedValue;
        
        // Check if any parameters have changed from their active values
        this.updateAdvancedRerunButtonState();
    }
    
    updateAdvancedRerunButtonState() {
        const rerunBtn = this.domElements.advancedRerunSearchBtn;
        if (!rerunBtn) return;
        
        // Check if any parameter has changed from its active value
        let hasChanges = false;
        for (const [paramId, currentValue] of Object.entries(this.currentAdvancedParams)) {
            const activeValue = this.activeAdvancedParams[paramId];
            if (currentValue !== activeValue) {
                hasChanges = true;
                break;
            }
        }
        
        rerunBtn.disabled = !hasChanges;
    }
    
    async rerunSearchWithAdvancedParameters() {
        
        // Save current values as the "active" values
        this.activeAdvancedParams = { ...this.currentAdvancedParams };
        
        // Disable the rerun button
        const rerunBtn = this.domElements.advancedRerunSearchBtn;
        if (rerunBtn) {
            rerunBtn.disabled = true;
        }
        
        // Track parameter update
        this.analytics.trackEvent('Advanced Parameters Updated', {
            'parameters': this.activeAdvancedParams,
            'has_active_search': this.searchResults.length > 0
        });
        
        // Show confirmation that search was rerun
        this.showAdvancedSettingsConfirmation();
        
        // Rerun the current search if we have one
        if (this.lastSearchRequestData) {
            await this.handleSearch();
        }
    }
    
    showAdvancedSettingsConfirmation() {
        // Create a temporary confirmation message
        const rerunBtn = this.domElements.advancedRerunSearchBtn;
        if (rerunBtn) {
            const originalText = rerunBtn.textContent;
            rerunBtn.textContent = 'Search Updated!';
            rerunBtn.style.background = 'rgba(29, 185, 84, 0.8)';
            
            setTimeout(() => {
                rerunBtn.textContent = originalText;
                rerunBtn.style.background = '';
            }, 2000);
        }
    }
    
    async initializeAdvancedSettingsValues() {
        // Fetch default parameters from backend
        try {
            const defaultParams = await this.api.get('/api/default_ranking_config');
            this.populateAdvancedSettingsForm(defaultParams);
            
            // Set both current and active to defaults initially
            this.currentAdvancedParams = { ...defaultParams };
            this.activeAdvancedParams = { ...defaultParams };
        } catch (error) {
            console.error('Error fetching default ranking config:', error);
            this.populateAdvancedSettingsFormWithDefaults();
        }
        
        this.updateAdvancedRerunButtonState();
    }
    
    populateAdvancedSettingsForm(params) {
        for (const [paramId, value] of Object.entries(params)) {
            const element = document.getElementById(paramId);
            if (element) {
                element.value = value;
            }
        }
    }
    
    populateAdvancedSettingsFormWithDefaults() {
        // Hardcoded fallback defaults matching RankingConfig
        const defaults = {
            'H_c': 30.0, 'H_E': 90.0, 'gamma_s': 1.2, 'gamma_f': 1.4, 'kappa': 1.5,
            'alpha_0': 3.0, 'beta_0': 3.0, 'K_s': 3.0, 'K_E': 10.0, 'gamma_A': 1.0,
            'eta': 1.2, 'tau': 0.7, 'beta_f': 1.5, 'K_life': 10.0, 'K_recent': 5.0,
            'psi': 1.4, 'k_neighbors': 50, 'sigma': 10.0, 'knn_embed_type': 'full_profile',
            'beta_p': 0.4, 'beta_s': 0.4, 'beta_a': 0.2, 'kappa_E': 0.25,
            'theta_c': 0.95, 'tau_c': 0.02, 'K_c': 8.0, 'tau_K': 2, 'M_A': 5.0,
            'K_fam': 9.0, 'R_min': 3.0, 'C_fam': 0.25, 'min_plays': 4,
            'beta_track': 0.5, 'beta_artist_pop': 0.15, 'beta_artist_personal': 0.0,
            'beta_genre': 0.2, 'beta_pop': 0.15
        };
        
        this.populateAdvancedSettingsForm(defaults);
        this.currentAdvancedParams = { ...defaults };
        this.activeAdvancedParams = { ...defaults };
    }
    
    async resetAdvancedParametersToDefaults() {
        
        try {
            // Fetch fresh defaults from API
            const defaults = await this.api.get('/api/default_ranking_config');
            
            // Populate form with defaults
            this.populateAdvancedSettingsForm(defaults);
            
            // Update current params but DON'T update active params 
            // (so the button will detect changes)
            this.currentAdvancedParams = { ...defaults };
            
            // Update button state - should enable the button if current != active
            this.updateAdvancedRerunButtonState();
        } catch (error) {
            console.error('Error fetching defaults for reset:', error);
            return;
        }
        
        // Track reset action
        this.analytics.trackEvent('Advanced Parameters Reset', {
            'reset_to_defaults': true
        });
    }
    
    getActiveAdvancedParams() {
        
        // If we don't have active params but have current params, use current params
        let paramsToUse = this.activeAdvancedParams;
        if (!paramsToUse || Object.keys(paramsToUse).length === 0) {
            paramsToUse = this.currentAdvancedParams;
        }
        
        // Return empty object if still no parameters
        if (!paramsToUse || Object.keys(paramsToUse).length === 0) {
            return {};
        }
        
        // Convert parameter names to match backend expectation
        const backendParams = {};
        for (const [key, value] of Object.entries(paramsToUse)) {
            // Most parameters can be sent directly, but handle special cases if needed
            backendParams[key] = value;
        }
        
        return backendParams;
    }
    
    showPersonalizationControls(hasHistory) {
        const controls = document.getElementById('personalization-controls');
        const resultsRight = document.querySelector('.results-right');
        const topArtistsFilterOption = document.getElementById('top-artists-filter-option');
        const advancedSettingsSection = document.getElementById('advanced-settings-section');
        
        if (controls && resultsRight) {
            this.hasPersonalizationHistory = hasHistory;
            
            if (hasHistory) {
                // Move personalization controls to results-right area
                controls.style.display = 'flex';
                resultsRight.appendChild(controls);
                
                // Initialize slider value positions
                this.initializeSliderPositions();
                
                // Show advanced settings section in history mode
                if (advancedSettingsSection) {
                    advancedSettingsSection.style.display = 'block';
                }
                
                // Initialize advanced parameters with defaults if not already done
                if (!this.currentAdvancedParams || Object.keys(this.currentAdvancedParams).length === 0) {
                    this.populateAdvancedSettingsFormWithDefaults();
                }
                
                // Hide top artists filter in history mode
                if (topArtistsFilterOption) {
                    topArtistsFilterOption.style.display = 'none';
                }
            } else {
                // Hide personalization controls in no-history mode
                controls.style.display = 'none';
                
                // Hide advanced settings section in no-history mode
                if (advancedSettingsSection) {
                    advancedSettingsSection.style.display = 'none';
                }
                
                // Show top artists filter in no-history mode
                if (topArtistsFilterOption) {
                    topArtistsFilterOption.style.display = 'block';
                }
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
    // Player will be initialized when needed in playSong() - no need to initialize here
}; 