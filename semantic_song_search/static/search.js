class SearchManager {
    constructor(app, api, analytics) {
        this.app = app;
        this.api = api;
        this.analytics = analytics;
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

    handleSearchTypeChange(searchType) {
        const suggestionsContainer = this.app.domElements.suggestions;
        const querySection = document.getElementById('query-section');
        const searchInput = this.app.domElements.searchInput;
        
        // Track search type change
        this.analytics.trackEvent('Search Type Changed', {
            'new_search_type': searchType,
            'previous_search_type': this.app.currentSearchType || 'unknown'
        });
        
        // Update current search type immediately
        this.app.currentSearchType = searchType;
        
        if (searchType === 'song') {
            searchInput.placeholder = "🔍 Search for a song or artist... (e.g., \"Espresso\", \"Sabrina Carpenter\")";
            this.app.resultsUIManager.clearResults();
        } else {
            searchInput.placeholder = "🔍 Describe the vibe and genre you're looking for... (e.g., \"playful summery pop\", \"motivational workout hip hop\")";
            suggestionsContainer.style.display = 'none';
            querySection.style.display = 'none';
            this.app.resultsUIManager.clearResults();
        }
    }
    
    
    async handleSearchInput(query) {
        const searchType = this.getSearchType();
        const suggestionsContainer = this.app.domElements.suggestions;
        
        if (searchType === 'song' && query.trim().length > 2) {
            try {
                const suggestions = await this.api.get(`/api/search_suggestions?query=${encodeURIComponent(query)}`);
                this.app.currentSuggestions = suggestions; // Store suggestions for Enter key handling
                this.displaySuggestions(suggestions);
                suggestionsContainer.style.display = 'block';
            } catch (error) {
                console.error('Error fetching suggestions:', error);
                this.app.currentSuggestions = []; // Clear suggestions on error
                suggestionsContainer.style.display = 'none';
            }
        } else {
            this.app.currentSuggestions = []; // Clear suggestions for text search or short queries
            suggestionsContainer.style.display = 'none';
        }
    }
    
    displaySuggestions(suggestions) {
        const container = this.app.domElements.suggestions;
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
        this.app.currentQuerySong = suggestion;
        this.app.domElements.searchInput.value = suggestion.label;
        this.app.domElements.suggestions.style.display = 'none';
        this.displayQueryCard(suggestion);
        this.handleSearch();
    }
    
    displayQueryCard(song) {
        const querySection = document.getElementById('query-section');
        const queryCard = document.getElementById('query-card');
        
        queryCard.dataset.spotifyId = song.spotify_id; // Set the spotify ID for consistency
        queryCard.innerHTML = this.app.resultsUIManager.createSongCardHTML(song, { isQuery: true });
        querySection.style.display = 'block';
    }
    
    handleEnterKey() {
        const searchType = this.getSearchType();
        const currentInput = this.app.domElements.searchInput.value.trim();
        
        // For song-to-song search, if we're typing a new query (input doesn't match current song)
        // and we have suggestions, auto-select the top one
        if (searchType === 'song') {
            const hasValidSuggestions = this.app.currentSuggestions.length > 0;
            const isTypingNewQuery = !this.app.currentQuerySong || 
                (this.app.currentQuerySong && this.app.currentQuerySong.label !== currentInput);
            
            if (hasValidSuggestions && isTypingNewQuery) {
                // Auto-select the first (top) suggestion
                const topSuggestion = this.app.currentSuggestions[0];
                this.selectSuggestion(topSuggestion);
                return; // selectSuggestion will call handleSearch() internally
            }
        }
        
        // For text search or when a song is already selected, proceed with normal search
        this.handleSearch();
    }
    
    async handleSearch() {
        const searchParams = this.extractSearchParameters();
        if (!searchParams.query) return;
        
        this.trackSearchInitiation(searchParams);
        this.app.searchCount++;
        
        // Handle top artists filter loading
        const filterTopArtists = await this.ensureTopArtistsLoaded(searchParams.filterTopArtists);
        
        this.prepareForNewSearch(searchParams);
        
        // Clear existing results immediately to show loading state
        if (this.app.searchResults.length > 0 || this.app.domElements.resultsGrid.children.length > 0) {
            this.app.domElements.resultsGrid.innerHTML = '';
            this.app.domElements.resultsHeader.style.display = 'none';
            this.app.domElements.loadMoreContainer.style.display = 'none';
        }
        
        this.app.resultsUIManager.showLoading(true);
        this.app.resultsUIManager.hideWelcomeMessage();
        
        try {
            const requestData = this.buildSearchRequest(searchParams);
            const data = await this.performSearch(requestData);
            this.processSearchResults(data);
            
        } catch (error) {
            console.error('Search error:', error);
            this.app.resultsUIManager.showError('Search failed. Please try again.');
        } finally {
            this.app.resultsUIManager.showLoading(false);
        }
    }
    
    
    extractSearchParameters() {
        const searchType = this.getSearchType();
        const query = this.app.domElements.searchInput.value.trim();
        const topArtistsFilter = this.app.domElements.topArtistsFilter;
        const filterTopArtists = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        return { searchType, query, filterTopArtists };
    }
    
    trackSearchInitiation(searchParams) {
        const { searchType, query, filterTopArtists } = searchParams;
        
        const searchProperties = {
            'search_type': searchType,
            'descriptors': 'all', // Using all descriptors simultaneously
            'is_filtered': filterTopArtists,
            'is_manual_selection': this.app.isManualSelectionMode,
            'selected_songs_count': this.app.selectedSongs.size,
            'has_spotify_auth': this.app.isAuthenticated,
            'has_results': this.app.searchResults.length > 0,
            'is_new_search': this.app.searchResultsId !== `${searchType}:all:${query}:${this.app.currentQuerySong?.song_idx || ''}`
        };
        
        // Add query-specific information
        if (searchType === 'text' && query) {
            searchProperties.query = query;
            searchProperties.query_length = query.length;
        } else if (searchType === 'song' && this.app.currentQuerySong) {
            searchProperties.has_query_song = true;
            searchProperties.query_song_idx = this.app.currentQuerySong.song_idx;
            searchProperties.query_song_name = this.app.currentQuerySong.song || '';
            searchProperties.query_artist_name = this.app.currentQuerySong.artist || '';
        }
        
        this.analytics.trackEvent('Search Initiated', searchProperties);
    }
    
    async ensureTopArtistsLoaded(filterTopArtists) {
        if (filterTopArtists && !this.app.topArtistsLoaded) {
            await this.app.loadTopArtists();
            // If loading failed, uncheck the filter and continue without it
            if (!this.app.topArtistsLoaded) {
                this.app.domElements.topArtistsFilter.checked = false;
                return false;
            }
        }
        return filterTopArtists;
    }
    
    prepareForNewSearch(searchParams) {
        const { searchType, query } = searchParams;
        
        // Create a search identifier based on actual search parameters
        const newSearchId = `${searchType}:all:${query}:${this.app.currentQuerySong?.song_idx || ''}`;
        const isNewSearch = this.app.searchResultsId !== newSearchId;
        
        // Reset pagination for new searches
        this.app.currentOffset = 0;
        this.app.searchResults = [];
        this.app.originalSearchResults = [];
        this.app.baseSearchResults = []; // Reset base dataset
        
        // Reset manual selection for new searches
        this.app.isManualSelectionMode = false;
        this.app.selectedSongs.clear();
        const manualSelectionToggle = this.app.domElements.manualSelectionToggle;
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        this.app.playlistExport.updateExportFormDisplay();
        
        // Reset queue when starting a new search
        if (isNewSearch) {
            this.app.searchResultsId = newSearchId;
            this.app.player.resetAutoPlayQueue();
        }
        
        // Store current search parameters for playlist name auto-population
        this.app.currentQuery = query;
        this.app.currentSearchType = searchType;
        
        return isNewSearch;
    }
    
    buildSearchRequest(searchParams) {
        const { searchType, query } = searchParams;

        // Get no-history weights
        const noHistoryWeights = this.app.personalizationManager.getActiveNoHistoryWeights();

        const requestData = {
            search_type: searchType,
            query: query,
            k: this.app.search_k,
            offset: 0
        };

        // Add personalization parameters only in history mode
        if (this.app.hasPersonalizationHistory) {
            requestData.lambda_val = this.app.activeLambdaVal !== undefined ? this.app.activeLambdaVal : (this.app.currentLambdaVal !== undefined ? this.app.currentLambdaVal : 0.5);
            requestData.familiarity_min = this.app.activeFamiliarityMin !== undefined ? this.app.activeFamiliarityMin : (this.app.currentFamiliarityMin !== undefined ? this.app.currentFamiliarityMin : 0.0);
            requestData.familiarity_max = this.app.activeFamiliarityMax !== undefined ? this.app.activeFamiliarityMax : (this.app.currentFamiliarityMax !== undefined ? this.app.currentFamiliarityMax : 1.0);
        }

        // Add no-history weights if we don't have personalization history
        if (!this.app.hasPersonalizationHistory && Object.keys(noHistoryWeights).length > 0) {
            Object.assign(requestData, noHistoryWeights);
        }

        if (searchType === 'song' && this.app.currentQuerySong) {
            requestData.song_idx = this.app.currentQuerySong.song_idx;
        }

        // Store current search data for load more functionality and weight updates
        this.app.currentSearchData = { ...requestData };
        this.app.lastSearchRequestData = { ...requestData };

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
        this.app.searchResults = data.results;
        this.app.originalSearchResults = [...data.results]; // Store original unfiltered results for client-side filtering
        this.app.baseSearchResults = [...data.results]; // Initialize base dataset (pre-artist-filter)
        this.app.currentOffset = data.pagination.offset + data.pagination.limit;
        this.app.totalResultsCount = data.pagination.total_count;
        this.app.hasMoreResults = data.pagination.has_more;
        this.app.isFiltered = false; // Reset since server doesn't filter anymore
        
        // Apply client-side filter if checkbox is currently checked
        const topArtistsFilter = this.app.domElements.topArtistsFilter;
        if (topArtistsFilter.checked && !topArtistsFilter.disabled && this.app.topArtistsLoaded) {
            this.applyClientSideFilter();
        } else {
            this.displayResults(data, false);
        }
    }
    
    async loadMoreResults() {
        if (this.app.isLoadingMore || !this.app.hasMoreResults || !this.app.currentSearchData) {
            return;
        }
        
        // Track load more click
        this.analytics.trackEvent('Load More Clicked', {
            'current_results_count': this.app.searchResults.length,
            'current_offset': this.app.currentOffset,
            'search_type': this.app.currentSearchData.search_type,
            'descriptors': 'all', // Using all descriptors simultaneously
            'has_filter_active': this.app.domElements.topArtistsFilter.checked,
            'is_manual_selection_mode': this.app.isManualSelectionMode
        });
        
        this.app.isLoadingMore = true;
        this.app.resultsUIManager.showLoadMoreLoading(true);
        
        try {
            const requestData = {
                ...this.app.currentSearchData,
                offset: this.app.currentOffset,
                // Use current personalization parameters (may have changed since original search)
                lambda_val: this.app.currentLambdaVal !== undefined ? this.app.currentLambdaVal : 0.5,
                familiarity_min: this.app.currentFamiliarityMin !== undefined ? this.app.currentFamiliarityMin : 0.0,
                familiarity_max: this.app.currentFamiliarityMax !== undefined ? this.app.currentFamiliarityMax : 1.0
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
            
            const previousResultsCount = this.app.searchResults.length;
            
            // Add new results to original unfiltered results
            this.app.originalSearchResults = [...this.app.originalSearchResults, ...data.results];
            this.app.currentOffset = data.pagination.offset + data.pagination.limit;
            this.app.hasMoreResults = data.pagination.has_more;

            // Auto-select new songs if manual selection is enabled
            if (this.app.isManualSelectionMode) {
                data.results.forEach(song => {
                    this.app.selectedSongs.add(song.song_idx);
                });
            }

            // Check if we have client-side filtering active
            const topArtistsFilter = this.app.domElements.topArtistsFilter;
            const hasTopArtistsFilter = topArtistsFilter.checked && !topArtistsFilter.disabled && this.app.topArtistsLoaded;
            const hasArtistFilter = this.app.artistFilterState.isActive;
            const hasAnyFilter = hasTopArtistsFilter || hasArtistFilter;

            // Update base dataset for non-filtered scenarios
            if (!hasAnyFilter) {
                // No filters active, base dataset = original results
                this.app.baseSearchResults = [...this.app.originalSearchResults];
            }

            console.log(`🎵 Load More: topArtists=${hasTopArtistsFilter}, artistFilter=${hasArtistFilter}, newResults=${data.results.length}`);

            if (hasAnyFilter) {
                // Use unified filtering approach for any active filters
                // This rebuilds searchResults from complete originalSearchResults and updates artist dropdown
                this.applyClientSideFilter();
            } else {
                // No filtering - just add new results and display
                this.app.searchResults = [...this.app.searchResults, ...data.results];
                this.displayResults(data, true);

                // Update manual selection state for newly loaded cards if needed
                if (this.app.isManualSelectionMode) {
                    this.app.resultsUIManager.updateAllCardSelections();
                    this.app.playlistExport.updateExportFormDisplay(); // Update export form with new selection count
                }
            }
            
            console.log('🎵 Added', data.results.length, 'new songs to queue (was', previousResultsCount, ')');
            
        } catch (error) {
            console.error('Load more error:', error);
            this.app.resultsUIManager.showError('Failed to load more results. Please try again.');
        } finally {
            this.app.isLoadingMore = false;
            this.app.resultsUIManager.showLoadMoreLoading(false);
        }
    }
    
    displayResults(data, isLoadMore = false, rebuildArtistDropdown = true) {
        const resultsHeader = this.app.domElements.resultsHeader;
        const searchInfo = document.getElementById('search-info');
        const resultsGrid = this.app.domElements.resultsGrid;
        const loadMoreContainer = this.app.domElements.loadMoreContainer;
        
        // Update header (show current results count with new formatting)
        this.app.resultsUIManager.updateResultsCount();
        
        // Create enhanced search info with ranking details
        let searchInfoText = `${data.search_type} search`;
        
        searchInfo.textContent = searchInfoText;
        resultsHeader.style.display = 'flex';
        
        // Display ranking weights if available (only for new searches)
        if (!isLoadMore) {
            this.app.personalizationManager.displayRankingWeights(data.ranking_weights);
            
            // Show/hide personalization controls based on history availability
            const hasHistory = data.ranking_weights && data.ranking_weights.has_history;
            this.app.personalizationManager.showPersonalizationControls(hasHistory);
        }
        
        // Clear results grid only for new searches, not for load more
        if (!isLoadMore) {
            resultsGrid.innerHTML = '';
            // Reset listener tracking for new searches
            this.app.resultsUIManager.resetEventListenerTracking();
        }
        
        // Handle empty results case
        if (!isLoadMore && this.app.searchResults.length === 0) {
            const message = this.app.isFiltered ? 
                'No songs found matching your filters. Try searching without filters or adding more listening history to Spotify.' :
                'No results found. Try adjusting your search terms or selecting a different embedding type.';
            
            resultsGrid.innerHTML = `
                <div class="no-results-message">
                    <h3>No results found</h3>
                    <p>${message}</p>
                </div>
            `;
            loadMoreContainer.style.display = 'none';
            this.app.domElements.exportSection.style.display = 'none';
            return;
        }
        
        // Show export section when results are available (only for new searches)
        if (!isLoadMore && this.app.searchResults.length > 0) {
            this.app.domElements.exportSection.style.display = 'block';
            // Update the song count input placeholder to show available results
            this.app.resultsUIManager.updateSongCountHint();
        }

        // Build or update artist filter data (performance-optimized)
        // Only rebuild dropdown if explicitly requested (not for filtered results display)
        if (rebuildArtistDropdown) {
            this.app.resultsUIManager.buildArtistDataMaps(this.app.baseSearchResults, isLoadMore);

            // Populate or update artist filter dropdown
            if (!isLoadMore || this.app.artistFilterState.artistTrackCounts.size > 0) {
                this.app.resultsUIManager.populateArtistFilterDropdown();
            }
        }

        // Add new results to the grid  
        const startIndex = isLoadMore ? this.app.searchResults.length - data.results.length : 0;
        data.results.forEach((song, index) => {
            const card = document.createElement('div');
            let cardClasses = 'song-card';
            
            // Add selection-related classes
            if (this.app.isManualSelectionMode) {
                cardClasses += ' selectable';
                if (this.app.selectedSongs.has(song.song_idx)) {
                    cardClasses += ' selected';
                }
            }
            
            card.className = cardClasses;
            card.dataset.spotifyId = song.spotify_id; // Set the spotify ID for the updatePlayingCards function
            card.innerHTML = this.app.resultsUIManager.createSongCardHTML(song, { 
                rank: startIndex + index + 1, 
                similarity: song.similarity,
                fieldValue: song.field_value,
                isSelected: this.app.selectedSongs.has(song.song_idx)
            });

            resultsGrid.appendChild(card);
        });
        
        // Ensure results container has correct CSS class for manual selection mode
        const resultsContainer = this.app.domElements.resultsContainer;
        if (this.app.isManualSelectionMode) {
            resultsContainer.classList.add('manual-selection-mode');
        } else {
            resultsContainer.classList.remove('manual-selection-mode');
        }
        
        // Attach event listeners for song cards (handles both normal and selection modes)
        // For load more, only attach listeners to new cards starting from startIndex
        const listenerStartIndex = isLoadMore ? startIndex : 0;
        this.app.resultsUIManager.attachSongCardEventListeners(listenerStartIndex);
        
        // Update load more button visibility
        this.app.resultsUIManager.updateLoadMoreButton();
        
        // Update song count hint if export section is visible
        const exportSection = this.app.domElements.exportSection;
        if (exportSection && exportSection.style.display !== 'none') {
            this.app.resultsUIManager.updateSongCountHint();
        }
    }

    applyClientSideFilter() {
        const topArtistsFilter = this.app.domElements.topArtistsFilter;
        const filterEnabled = topArtistsFilter.checked && !topArtistsFilter.disabled;
        
        
        if (!filterEnabled || !this.app.topArtistsLoaded || !this.app.topArtists || this.app.topArtists.length === 0) {
            // Use all original results as base dataset
            console.log(`🔍 Using all ${this.app.originalSearchResults.length} original results as base dataset (Top Artists disabled)`);
            this.app.baseSearchResults = [...this.app.originalSearchResults];
            this.app.isFiltered = false;
        } else {
            // Filter to only show songs by top artists and update base dataset
            const topArtistsSet = new Set(this.app.topArtists.map(artist => artist.name.toLowerCase()));
            console.log(`🔍 Filtering with top artists:`, Array.from(topArtistsSet).slice(0, 5), '...');

            const filteredOut = [];
            this.app.baseSearchResults = this.app.originalSearchResults.filter(song => {
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

            this.app.isFiltered = true;
        }

        // Apply artist filter on top of the base dataset (if active)
        const artistFilter = this.app.artistFilterState;
        if (artistFilter.isActive && artistFilter.selectedArtists.size > 0 &&
            artistFilter.selectedArtists.size < artistFilter.artistTrackCounts.size) {
            console.log(`🎯 Applying artist filter on top of base dataset`);

            const selectedArtistsSet = new Set(artistFilter.selectedArtists);
            this.app.searchResults = this.app.baseSearchResults.filter(song => {
                const artists = song.all_artists && song.all_artists.length > 0
                    ? song.all_artists
                    : [song.artist].filter(artist => artist);

                return artists.some(artist => {
                    if (!artist || typeof artist !== 'string') return false;
                    return selectedArtistsSet.has(artist.trim());
                });
            });
            this.app.isFiltered = true;
        } else {
            // No artist filter active, use base dataset as final results
            this.app.searchResults = [...this.app.baseSearchResults];
        }
        
        // Remove filtered-out songs from manual selection
        this.cleanupManualSelection();

        // Update the display with filtered results
        const mockData = {
            results: this.app.searchResults,
            search_type: this.app.currentSearchData?.search_type || 'text',
            query: this.app.currentSearchData?.query || '',
            pagination: {
                offset: 0,
                limit: this.app.searchResults.length,
                total_count: this.app.isFiltered ? null : this.app.originalSearchResults.length,
                has_more: false, // No more results since we're showing all filtered results
                is_filtered: this.app.isFiltered,
                returned_count: this.app.searchResults.length
            }
        };

        this.displayResults(mockData, false);
        
        // Sync manual selection state after filtering re-creates the cards
        if (this.app.isManualSelectionMode) {
            this.app.resultsUIManager.updateAllCardSelections();
            this.app.playlistExport.updateExportFormDisplay(); // Update export form with current selection count
            this.app.resultsUIManager.updateResultsCount(); // Update results count display
        }
    }

    // Performance-optimized artist filter logic
    applyArtistFilter() {
        /**
         * Apply artist filter using performance-optimized Set operations
         * Caches filtered results to avoid re-filtering unchanged data
         */
        const artistFilter = this.app.artistFilterState;

        // If no filter is active or no artists selected, show all base results
        if (!artistFilter.isActive || artistFilter.selectedArtists.size === 0) {
            this.app.searchResults = [...this.app.baseSearchResults];
            this.app.isFiltered = false;
            this.cleanupManualSelection();
            return;
        }

        // Check if we can use cached filtered results
        if (artistFilter.filteredResults &&
            this.setsAreEqual(artistFilter.selectedArtists, artistFilter.lastFilteredWith)) {
            this.app.searchResults = artistFilter.filteredResults;
            this.app.isFiltered = true;
            this.cleanupManualSelection();
            return;
        }

        // Perform efficient filtering using Set for O(1) artist lookups
        const selectedArtistsSet = new Set(artistFilter.selectedArtists);

        const filteredResults = this.app.baseSearchResults.filter(song => {
            // Get all artists for this song
            const artists = song.all_artists && song.all_artists.length > 0
                ? song.all_artists
                : [song.artist].filter(artist => artist);

            // Include song if ANY of its artists are selected (O(k) where k = artists per song)
            return artists.some(artist => {
                if (!artist || typeof artist !== 'string') return false;
                return selectedArtistsSet.has(artist.trim());
            });
        });

        // Cache the filtered results for performance
        artistFilter.filteredResults = filteredResults;
        artistFilter.lastFilteredWith = new Set(artistFilter.selectedArtists);

        this.app.searchResults = filteredResults;
        this.app.isFiltered = true;

        // Remove filtered-out songs from manual selection
        this.cleanupManualSelection();

        console.log(`🎯 Artist filter applied: ${filteredResults.length}/${this.app.baseSearchResults.length} songs match ${selectedArtistsSet.size} selected artists`);
    }

    applyArtistFilterAndUpdate() {
        /**
         * Apply artist filter and update the display
         * This is called when the "Apply" button is clicked
         */
        const artistFilter = this.app.artistFilterState;

        // Update the active state
        artistFilter.isActive = artistFilter.selectedArtists.size < artistFilter.artistTrackCounts.size;

        // Apply the filter
        this.applyArtistFilter();

        // Update the display
        this.updateDisplayAfterFilter();

        // Update original selected state to reflect applied changes
        artistFilter.originalSelectedArtists = new Set(artistFilter.selectedArtists);

        // Update button states
        this.app.resultsUIManager.updateApplyButtonState();
        this.app.resultsUIManager.updateArtistFilterButtonText();
    }

    updateDisplayAfterFilter() {
        /**
         * Update the UI display after applying artist filter
         * This is used for full filter application (not Load More)
         */
        // Create mock data structure for displayResults
        const mockData = {
            results: this.app.searchResults,
            search_type: this.app.currentSearchData?.search_type || 'text',
            query: this.app.currentSearchData?.query || '',
            pagination: {
                offset: 0,
                limit: this.app.searchResults.length,
                total_count: this.app.artistFilterState.isActive ? null : this.app.originalSearchResults.length,
                has_more: false, // No more results when filtering from Apply button
                is_filtered: this.app.isFiltered,
                returned_count: this.app.searchResults.length
            }
        };

        this.displayResults(mockData, false, false); // Don't rebuild artist dropdown

        // Sync manual selection state if active
        if (this.app.isManualSelectionMode) {
            this.app.resultsUIManager.updateAllCardSelections();
            this.app.playlistExport.updateExportFormDisplay();
            this.app.resultsUIManager.updateResultsCount();
        }
    }

    cleanupManualSelection() {
        /**
         * Remove filtered-out songs from manual selection to maintain consistency
         * Only selected songs that are currently visible should remain selected
         */
        if (this.app.isManualSelectionMode && this.app.selectedSongs.size > 0) {
            // Create a set of currently visible song indices for efficient lookup
            const visibleSongIndices = new Set(this.app.searchResults.map(song => song.song_idx));

            // Remove any selected songs that are no longer visible
            const originalSelectionSize = this.app.selectedSongs.size;
            for (const songIdx of this.app.selectedSongs) {
                if (!visibleSongIndices.has(songIdx)) {
                    this.app.selectedSongs.delete(songIdx);
                }
            }

            const removedCount = originalSelectionSize - this.app.selectedSongs.size;
            if (removedCount > 0) {
                console.log(`🎯 Removed ${removedCount} filtered-out songs from selection (${this.app.selectedSongs.size} songs still selected)`);
            }
        }
    }

    setsAreEqual(set1, set2) {
        /**
         * Efficiently compare two sets for equality
         * Duplicated from ResultsUIManager for performance (avoid cross-class calls)
         */
        if (set1.size !== set2.size) return false;
        for (const item of set1) {
            if (!set2.has(item)) return false;
        }
        return true;
    }
}
