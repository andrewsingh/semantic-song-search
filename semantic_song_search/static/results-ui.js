class ResultsUIManager {
    constructor(app, api, analytics) {
        this.app = app;
        this.api = api;
        this.analytics = analytics;
    }

    createSongCardHTML(song, options = {}) {
        const { rank, similarity, isQuery = false, fieldValue = null, isSelected = false } = options;

        let tagsHTML = '';
        let playButtonHTML = '';
        if (!isQuery && song.spotify_id) {
            playButtonHTML = `<button class="song-play-btn" title="Play song">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
            </button>`;

            // Get tags and genres from song data
            const { tags } = this.formatTagsGenresFromSong(song);

            // Show first 3 tags
            const visibleTags = tags.slice(0, 3);
            const tagsElements = visibleTags.map(tag =>
                `<span class="tag-item">${escapeHtml(tag)}</span>`
            ).join('');

            tagsHTML = `
                <div class="card-tags">
                    <div class="tags-container">${tagsElements}</div>
                    ${playButtonHTML}
                </div>
            `;
        }
        
        // Restore accordion section for tags and genres
        let accordionHTML = '';
        if (!isQuery) {
            const tagsGenresObj = this.formatTagsGenresFromSong(song);
            const accordionContent = this.formatTagsGenresForDisplay(tagsGenresObj);

            // Only show accordion if there are tags or genres to display
            if (tagsGenresObj.tags.length > 0 || tagsGenresObj.genres.length > 0) {
                accordionHTML = `
                    <div class="card-accordion">
                        <button class="accordion-toggle" aria-expanded="false">
                            <span class="accordion-title">Tags & Genres</span>
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
            if (song.components || song.scoring_components) {
                const components = song.components || song.scoring_components;
                
                scoringComponentsHTML = `
                    <div class="scoring-components">
                        <span class="scoring-component" title="Song descriptor similarity">
                            <span class="component-label">S:</span>
                            <span class="component-value">${((components.S_song * 100) || 0).toFixed(1)}</span>
                        </span>
                        <span class="scoring-component" title="Artist descriptor similarity">
                            <span class="component-label">A:</span>
                            <span class="component-value">${((components.S_artist * 100) || 0).toFixed(1)}</span>
                        </span>
                        <span class="scoring-component" title="Total streams score">
                            <span class="component-label">T:</span>
                            <span class="component-value">${((components.S_streams_total * 100) || 0).toFixed(1)}</span>
                        </span>
                        <span class="scoring-component" title="Daily streams score">
                            <span class="component-label">D:</span>
                            <span class="component-value">${((components.S_streams_daily * 100) || 0).toFixed(1)}</span>
                        </span>
                        <span class="scoring-component" title="Release date similarity">
                            <span class="component-label">R:</span>
                            <span class="component-value">${((components.S_release_date * 100) || 0).toFixed(1)}</span>
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

        // Format multiple artists if available, fallback to single artist
        const artistText = song.all_artists && song.all_artists.length > 1 
            ? song.all_artists.join(', ')
            : song.artist;

        return `
            ${checkboxHTML}
            <div class="card-header">
                <img src="${escapeHtml(song.cover_url || '')}" alt="Cover" class="card-cover">
                <div class="card-info">
                    <div class="card-title">${escapeHtml(song.song)}</div>
                    <div class="card-artist">${escapeHtml(artistText)}</div>
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
            .map(item => String(item).trim().toLowerCase()) // Convert to string and trim and lowercase
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
        this.app.domElements.resultsHeader.style.display = 'none';
        this.app.domElements.resultsGrid.innerHTML = '';
        this.app.domElements.loadMoreContainer.style.display = 'none';
        document.getElementById('welcome-message').style.display = 'block';
        this.app.domElements.exportSection.style.display = 'none';
        
        // Reset export accordion state
        const accordion = document.querySelector('.export-accordion');
        if (accordion) {
            accordion.classList.remove('expanded');
        }
        this.app.playlistExport.hideExportStatus();
        
        this.app.currentQuerySong = null;
        this.app.currentSuggestions = []; // Clear stored suggestions
        this.app.searchResults = [];
        this.app.originalSearchResults = [];
        this.app.baseSearchResults = []; // Clear base dataset
        this.app.currentSearchData = null;
        this.app.lastSearchRequestData = null;
        this.app.currentOffset = 0;
        this.app.totalResultsCount = 0;
        this.app.hasMoreResults = false;
        this.app.isFiltered = false;
        
        // Reset manual selection
        this.app.isManualSelectionMode = false;
        this.app.selectedSongs.clear();
        const manualSelectionToggle = this.app.domElements.manualSelectionToggle;
        if (manualSelectionToggle) {
            manualSelectionToggle.checked = false;
        }
        // Update export form to show number input instead of selection info
        this.app.playlistExport.updateExportFormDisplay();
        
        // Reset artist filter state
        this.resetArtistFilter();

        // Reset event listener tracking
        this.resetEventListenerTracking();

        // Reset auto-play queue
        this.app.player.resetAutoPlayQueue();
    }

    resetArtistFilter() {
        /**
         * Reset artist filter state for new searches
         */
        const artistFilter = this.app.artistFilterState;

        // Clear all data structures
        artistFilter.selectedArtists.clear();
        artistFilter.originalSelectedArtists.clear();
        artistFilter.artistTrackCounts.clear();
        artistFilter.artistToTracks.clear();
        artistFilter.filteredResults = null;
        artistFilter.lastFilteredWith = null;
        artistFilter.isActive = false;
        artistFilter.isDropdownOpen = false;

        // Clear dropdown content
        if (this.app.domElements.artistFilterList) {
            this.app.domElements.artistFilterList.innerHTML = '';
        }

        // Disable dropdown and reset button text
        const dropdownBtn = this.app.domElements.artistFilterDropdownBtn;
        if (dropdownBtn) {
            dropdownBtn.disabled = true;
            const buttonText = dropdownBtn.querySelector('.dropdown-filter-text');
            if (buttonText) {
                buttonText.textContent = 'Filter by Artist';
            }
        }

        // Hide dropdown
        if (this.app.domElements.artistFilterDropdown) {
            this.app.domElements.artistFilterDropdown.style.display = 'none';
        }
    }
    
    resetEventListenerTracking() {
        // This method doesn't need to do anything since we clear the DOM
        // but it's here for clarity and potential future use
    }
    
    updateLoadMoreButton() {
        const loadMoreContainer = this.app.domElements.loadMoreContainer;
        const loadMoreBtn = this.app.domElements.loadMoreBtn;
        
        if (this.app.hasMoreResults && this.app.searchResults.length > 0) {
            loadMoreContainer.style.display = 'block';
            loadMoreBtn.disabled = this.app.isLoadingMore;
            loadMoreBtn.textContent = this.app.isLoadingMore ? 'Loading...' : 'Load More Results';
        } else {
            loadMoreContainer.style.display = 'none';
        }
    }
    
    showLoadMoreLoading() {
        this.updateLoadMoreButton();
    }
    
    toggleManualSelection(enabled) {
        
        this.app.isManualSelectionMode = enabled;
        const resultsContainer = this.app.domElements.resultsContainer;
        
        if (enabled) {
            // Select all current songs by default
            this.app.selectedSongs.clear();
            this.app.searchResults.forEach((song) => {
                this.app.selectedSongs.add(song.song_idx);
            });
            
            // Show checkboxes and enable selection styling with CSS class
            resultsContainer.classList.add('manual-selection-mode');
            this.updateAllCardSelections();
        } else {
            // Clear selections when disabled
            this.app.selectedSongs.clear();
            
            // Hide checkboxes and disable selection styling
            resultsContainer.classList.remove('manual-selection-mode');
            this.clearAllCardSelections();
        }
        
        // Update results count display to show/hide selection count
        this.updateResultsCount();
        
        // Update export form display
        this.app.playlistExport.updateExportFormDisplay();
        
        // Event listeners don't need to be re-attached - they handle both modes dynamically
    }
    
    updateAllCardSelections() {
        const resultsGrid = this.app.domElements.resultsGrid;
        const songCards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        
        songCards.forEach((card, index) => {
            if (index < this.app.searchResults.length) {
                const song = this.app.searchResults[index];
                const isSelected = this.app.selectedSongs.has(song.song_idx);
                
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
        const resultsGrid = this.app.domElements.resultsGrid;
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
        const resultsGrid = this.app.domElements.resultsGrid;
        const songCards = resultsGrid.querySelectorAll('.song-card:not(.query-card)');
        
        // Only attach listeners to cards starting from startIndex to avoid duplicates
        for (let index = startIndex; index < songCards.length && index < this.app.searchResults.length; index++) {
            const card = songCards[index];
            const song = this.app.searchResults[index];
            
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
                
                if (this.app.isManualSelectionMode) {
                    // Manual selection mode: single click toggles selection
                    this.toggleSongSelection(song.song_idx, index);
                } else {
                    // Normal mode: single click plays
                    this.app.player.playSong(song);
                }
            });
            
            // Handle play button clicks (always plays regardless of mode)
            const playButton = card.querySelector('.song-play-btn');
            if (playButton) {
                playButton.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent card click
                    this.app.player.playSong(song);
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

            // Handle accordion toggle clicks
            const accordionToggle = card.querySelector('.accordion-toggle');
            if (accordionToggle) {
                accordionToggle.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent song play when clicking accordion
                    this.toggleAccordion(accordionToggle);
                });
            }
        }
    }
    
    
    updateResultsCount() {
        const resultsCount = this.app.domElements.resultsCount;
        if (!resultsCount) return;
        
        let resultsText = `${this.app.searchResults.length}`;
        
        // Add "filtered" if any filters are active
        if (this.app.isFiltered) {
            resultsText += ' filtered';
        }
        
        resultsText += ' results';
        
        // Add selected count if manual selection is active
        if (this.app.isManualSelectionMode) {
            const selectedCount = this.app.selectedSongs.size;
            resultsText += ` (${selectedCount} selected)`;
        }
        
        resultsCount.textContent = resultsText;
    }
    
    
    toggleSongSelection(songIdx, cardIndex) {
        if (this.app.selectedSongs.has(songIdx)) {
            this.app.selectedSongs.delete(songIdx);
        } else {
            this.app.selectedSongs.add(songIdx);
        }
        
        // Update the card's visual state
        this.updateSongCardSelection(cardIndex, this.app.selectedSongs.has(songIdx));
        
        
        // Update results count display to reflect new selection
        this.updateResultsCount();
        
        // Update export form display to reflect new selection count
        this.app.playlistExport.updateExportFormDisplay();
    }
    
    updateSongCardSelection(cardIndex, isSelected) {
        const resultsGrid = this.app.domElements.resultsGrid;
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

    updateSongCountHint() {
        const songCountInput = document.getElementById('song-count');
        if (songCountInput) {
            let availableText;
            if (this.app.isFiltered) {
                // When filtering is active
                availableText = this.app.hasMoreResults ? 
                    `${this.app.searchResults.length} loaded (filtered)` :
                    `${this.app.searchResults.length} available (filtered)`;
            } else {
                // Normal case - just show current count, with "more available" if applicable
                availableText = this.app.hasMoreResults ? 
                    `${this.app.searchResults.length} loaded, more available` :
                    `${this.app.searchResults.length} available`;
            }
            
            songCountInput.title = `Currently ${availableText}`;
            
            const label = document.querySelector('label[for="song-count"]');
            if (label) {
                label.textContent = `Number of Songs (${availableText}):`;
            }
        }
    }

    // Performance-optimized artist data extraction
    buildArtistDataMaps(songs, isIncremental = false) {
        /**
         * Efficiently extract artist data from search results using O(1) hashmaps
         * Always uses the provided songs array for track count calculations
         * @param {Array} songs - Array of song objects to build artist data from (should be baseSearchResults for accurate counts)
         * @param {boolean} isIncremental - Whether this is an incremental update (preserves existing selections)
         */
        const artistFilter = this.app.artistFilterState;

        // Store current selections for incremental updates
        let previouslySelectedArtists = new Set();
        if (isIncremental) {
            console.log('🎯 Load More - preserving existing selections and rebuilding from passed dataset');
            // Store current selections to preserve them
            previouslySelectedArtists = new Set(artistFilter.selectedArtists);
        }

        // Always clear maps and rebuild from passed songs parameter
        artistFilter.artistTrackCounts.clear();
        artistFilter.artistToTracks.clear();

        // Single pass through songs to build both maps simultaneously
        songs.forEach((song) => {
            // Get all artists for this song (handle both single artist and multiple artists)
            const artists = song.all_artists && song.all_artists.length > 0
                ? song.all_artists
                : [song.artist].filter(artist => artist); // Filter out null/undefined

            // Process each artist for this song
            artists.forEach(artist => {
                if (!artist || typeof artist !== 'string') return;

                const artistName = artist.trim();
                if (!artistName) return;

                // Update track count (O(1))
                const currentCount = artistFilter.artistTrackCounts.get(artistName) || 0;
                artistFilter.artistTrackCounts.set(artistName, currentCount + 1);

                // Update artist-to-tracks mapping (O(1))
                if (!artistFilter.artistToTracks.has(artistName)) {
                    artistFilter.artistToTracks.set(artistName, new Set());
                }
                artistFilter.artistToTracks.get(artistName).add(song.song_idx);
            });
        });

        // Handle artist selection state for incremental updates
        if (isIncremental) {
            // Start with previous selections, then add any new artists as selected (default behavior)
            artistFilter.selectedArtists = new Set(previouslySelectedArtists);

            // Add any new artists as selected by default (they're currently being shown)
            artistFilter.artistTrackCounts.forEach((_, artistName) => {
                if (!previouslySelectedArtists.has(artistName)) {
                    artistFilter.selectedArtists.add(artistName);
                    console.log(`🎯 New artist "${artistName}" added as selected (Load More)`);
                }
            });

            // Update original selected state to reflect new additions
            artistFilter.originalSelectedArtists = new Set(artistFilter.selectedArtists);
        }

        // Enable artist filter dropdown once we have data
        const dropdownBtn = this.app.domElements.artistFilterDropdownBtn;
        if (dropdownBtn && artistFilter.artistTrackCounts.size > 0) {
            dropdownBtn.disabled = false;
        }

        console.log(`🎯 Built artist maps: ${artistFilter.artistTrackCounts.size} unique artists, ${songs.length} songs processed`);
    }

    populateArtistFilterDropdown() {
        /**
         * Populate the artist filter dropdown with checkboxes and live counts
         * Uses performance-optimized data structures for O(1) operations
         */
        const artistFilter = this.app.artistFilterState;
        const dropdown = this.app.domElements.artistFilterList;

        if (!dropdown) {
            console.error('Artist filter dropdown element not found');
            return;
        }

        // Initialize original state for new population (before rendering checkboxes!)
        if (artistFilter.originalSelectedArtists.size === 0) {
            this.resetArtistFilterState();
        }

        // Clear existing content
        dropdown.innerHTML = '';

        // Get sorted artists by track count (descending) for better UX
        const sortedArtists = Array.from(artistFilter.artistTrackCounts.entries())
            .sort((a, b) => b[1] - a[1]); // Sort by count descending

        if (sortedArtists.length === 0) {
            dropdown.innerHTML = '<div class="dropdown-filter-empty">No artists found</div>';
            return;
        }

        // Add master "Select All" checkbox at the top
        const masterCheckboxItem = document.createElement('div');
        masterCheckboxItem.className = 'dropdown-filter-item';
        masterCheckboxItem.innerHTML = `
            <label class="dropdown-checkbox-label">
                <input type="checkbox" class="dropdown-checkbox master-select-checkbox"
                       data-action="select-all" checked>
                <span class="dropdown-artist-name"><strong>Select All / None</strong></span>
                <span class="dropdown-track-count">(${sortedArtists.length} artists)</span>
            </label>
        `;
        dropdown.appendChild(masterCheckboxItem);

        // Create checkboxes for each artist
        sortedArtists.forEach(([artistName, trackCount]) => {
            const isSelected = artistFilter.selectedArtists.has(artistName);

            const checkboxItem = document.createElement('div');
            checkboxItem.className = 'dropdown-filter-item';
            checkboxItem.innerHTML = `
                <label class="dropdown-checkbox-label">
                    <input type="checkbox" class="dropdown-checkbox"
                           data-artist="${artistName.replace(/"/g, '&quot;')}"
                           ${isSelected ? 'checked' : ''}>
                    <span class="dropdown-artist-name">${escapeHtml(artistName)}</span>
                    <span class="dropdown-track-count">(${trackCount})</span>
                </label>
            `;

            dropdown.appendChild(checkboxItem);
        });

        // Update dropdown button text to show total artists
        this.updateArtistFilterButtonText();
    }

    resetArtistFilterState() {
        /**
         * Reset artist filter to "all selected" state
         * Called when populating dropdown for new search results
         */
        const artistFilter = this.app.artistFilterState;

        console.log(`🎯 Resetting artist filter - selecting all ${artistFilter.artistTrackCounts.size} artists by default`);

        // Select all artists by default
        artistFilter.selectedArtists.clear();
        artistFilter.artistTrackCounts.forEach((_, artistName) => {
            artistFilter.selectedArtists.add(artistName);
        });

        // Store original state for change detection
        artistFilter.originalSelectedArtists = new Set(artistFilter.selectedArtists);

        // Update Apply button state
        this.updateApplyButtonState();
    }

    updateApplyButtonState() {
        /**
         * Enable/disable Apply button based on whether selection has changed
         * Uses efficient Set operations for O(1) comparison
         */
        const artistFilter = this.app.artistFilterState;
        const applyBtn = this.app.domElements.artistFilterApplyBtn;

        if (!applyBtn) return;

        // Check if current selection differs from original using Set operations
        const hasChanges = !this.setsAreEqual(artistFilter.selectedArtists, artistFilter.originalSelectedArtists);

        console.log(`🎯 Apply button ${hasChanges ? 'enabled' : 'disabled'} (${artistFilter.selectedArtists.size}/${artistFilter.originalSelectedArtists.size} selected)`);

        applyBtn.disabled = !hasChanges;
        applyBtn.textContent = hasChanges ? 'Apply' : 'Apply';
    }

    setsAreEqual(set1, set2) {
        /**
         * Efficiently compare two sets for equality using O(n) Set operations
         * @param {Set} set1 - First set
         * @param {Set} set2 - Second set
         * @returns {boolean} - Whether sets contain identical elements
         */
        if (set1.size !== set2.size) return false;

        for (const item of set1) {
            if (!set2.has(item)) return false;
        }

        return true;
    }

    handleArtistCheckboxChange(artistName, isChecked) {
        /**
         * Handle individual artist checkbox state change
         * Uses Set operations for O(1) updates
         */
        const artistFilter = this.app.artistFilterState;

        console.log(`🎯 Artist "${artistName}" ${isChecked ? 'selected' : 'deselected'}`);

        if (isChecked) {
            artistFilter.selectedArtists.add(artistName);
        } else {
            artistFilter.selectedArtists.delete(artistName);
        }

        // Update Apply button state based on changes
        this.updateApplyButtonState();
    }

    updateArtistFilterButtonText() {
        /**
         * Update the dropdown button text to show filter status
         */
        const artistFilter = this.app.artistFilterState;
        const buttonText = this.app.domElements.artistFilterDropdownBtn?.querySelector('.dropdown-filter-text');

        if (!buttonText) return;

        const totalArtists = artistFilter.artistTrackCounts.size;
        const selectedCount = artistFilter.selectedArtists.size;

        if (artistFilter.isActive && selectedCount < totalArtists) {
            buttonText.textContent = `Filter by Artist (${selectedCount}/${totalArtists})`;
        } else {
            buttonText.textContent = 'Filter by Artist';
        }
    }

    selectAllArtists() {
        /**
         * Select all artists in the current dropdown
         * Triggered by master "Select All" checkbox
         */
        const artistFilter = this.app.artistFilterState;

        console.log('🎯 Select All: Adding all artists to selection');

        // Add all current artists to selected set
        artistFilter.artistTrackCounts.forEach((_, artistName) => {
            artistFilter.selectedArtists.add(artistName);
        });

        // Update all individual checkboxes to checked state
        this.updateAllArtistCheckboxes(true);

        // Update Apply button state
        this.updateApplyButtonState();
    }

    unselectAllArtists() {
        /**
         * Unselect all artists in the current dropdown
         * Triggered by master "Select All" checkbox
         */
        const artistFilter = this.app.artistFilterState;

        console.log('🎯 Unselect All: Clearing all artist selections');

        // Clear all selected artists
        artistFilter.selectedArtists.clear();

        // Update all individual checkboxes to unchecked state
        this.updateAllArtistCheckboxes(false);

        // Update Apply button state
        this.updateApplyButtonState();
    }

    updateAllArtistCheckboxes(checked) {
        /**
         * Update all individual artist checkboxes to specified checked state
         * @param {boolean} checked - Whether checkboxes should be checked or unchecked
         */
        const dropdown = this.app.domElements.artistFilterList;
        if (!dropdown) return;

        // Update all individual artist checkboxes (not the master checkbox)
        const artistCheckboxes = dropdown.querySelectorAll('.dropdown-checkbox:not(.master-select-checkbox)');
        artistCheckboxes.forEach(checkbox => {
            checkbox.checked = checked;
        });
    }
}
