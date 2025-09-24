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
        
        // Reset event listener tracking
        this.resetEventListenerTracking();
        
        // Reset auto-play queue
        this.app.player.resetAutoPlayQueue();
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
    
    showLoadMoreLoading(show) {
        this.updateLoadMoreButton();
    }
    
    toggleManualSelection(enabled) {
        
        this.app.isManualSelectionMode = enabled;
        const resultsContainer = this.app.domElements.resultsContainer;
        
        if (enabled) {
            // Select all current songs by default
            this.app.selectedSongs.clear();
            this.app.searchResults.forEach((song, index) => {
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
}
