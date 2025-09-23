// Playlist Export Manager
// Handles all playlist creation functionality

class PlaylistExportManager {
    constructor(app, api, analytics) {
        this.app = app;
        this.api = api;
        this.analytics = analytics;
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
        if (!this.app.isAuthenticated || !this.app.accessToken) {
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
        
        if (this.app.isManualSelectionMode) {
            // In manual selection mode, use the number of selected songs
            songCount = this.app.selectedSongs.size;
        } else {
            // In normal mode, validate the song count input
            songCount = parseInt(songCountInput.value);
            
            if (isNaN(songCount) || songCount < 1 || songCount > 100) {
                this.showExportStatus('Number of songs must be between 1 and 100.', 'error');
                return { isValid: false };
            }
            
            // Additional check for extremely large requests when auto-loading is involved
            if (songCount > this.app.searchResults.length && songCount > 50 && this.app.hasMoreResults) {
                const proceed = confirm(
                    `You requested ${songCount} songs but only ${this.app.searchResults.length} are currently loaded.\n\n` +
                    `This will automatically load more results, which may take some time.\n\n` +
                    `Continue with auto-loading?`
                );
                if (!proceed) {
                    return { isValid: false };
                }
            }
        }
        
        if (!this.app.searchResults || this.app.searchResults.length === 0) {
            this.showExportStatus('No search results available to export.', 'error');
            return { isValid: false };
        }
        
        // Additional validation for manual selection mode
        if (this.app.isManualSelectionMode && this.app.selectedSongs.size === 0) {
            this.showExportStatus('No songs selected. Please check at least one song to export.', 'error');
            return { isValid: false };
        }
        
        // Check authentication
        if (!this.app.accessToken) {
            this.showExportStatus('Please <a href="/login" style="color: #1ed760; text-decoration: underline;">login to Spotify</a> first.', 'error');
            return { isValid: false };
        }
        
        return { isValid: true, playlistName, songCount };
    }
    
    getExportElements() {
        return {
            exportBtn: this.app.domElements.exportBtn,
            exportStatus: this.app.domElements.exportStatus,
            exportButtonText: document.querySelector('.export-button-text'),
            exportButtonLoading: document.querySelector('.export-button-loading')
        };
    }
    
    async prepareSongsForExport(songCount) {
        // Check if we need to load more results
        if (songCount > this.app.searchResults.length && this.app.hasMoreResults) {
            // Show loading message
            this.showExportStatus(`Loading more results to reach ${songCount} songs...`, 'info');
            
            // Auto-load more results until we have enough
            await this.autoLoadMoreForExport(songCount);
        }
        
        // Prepare song IDs for export
        let songsToExport;
        if (this.app.isManualSelectionMode) {
            // In manual selection mode, export only selected songs
            const selectedSongsInResults = this.app.searchResults.filter(song => this.app.selectedSongs.has(song.song_idx));
            
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
            songsToExport = this.app.searchResults.slice(0, songCount);
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
            if (spotifyIds.length < this.app.searchResults.length) {
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
            search_type: this.app.currentSearchType,
            descriptors: 'all', // Using all descriptors simultaneously
            is_filtered: this.app.isFiltered,
            is_manual_selection: this.app.isManualSelectionMode,
            selected_songs_count: this.app.selectedSongs.size
        };
        
        // Add query-specific context
        if (this.app.currentSearchType === 'text' && this.app.currentQuery) {
            searchContext.query = this.app.currentQuery;
            searchContext.query_length = this.app.currentQuery.length;
        } else if (this.app.currentSearchType === 'song' && this.app.currentQuerySong) {
            searchContext.query_song_idx = this.app.currentQuerySong.song_idx;
            searchContext.query_song_name = this.app.currentQuerySong.song || '';
            searchContext.query_artist_name = this.app.currentQuerySong.artist || '';
        }
        
        return searchContext;
    }
    
    handlePlaylistCreationSuccess(data, songCount) {
        const requestedText = data.songs_added < songCount ? 
            ` (${songCount} requested)` : '';
        const message = `
            ✅ Playlist created successfully!<br>
            <strong>${data.playlist_name}</strong><br>
            ${data.songs_added} tracks added${requestedText}<br>
            <a href="${data.playlist_url}" target="_blank" style="color: #1ed760; text-decoration: underline; font-weight: bold;">
                🎵 Open in Spotify ↗
            </a>
        `;
        this.showExportStatus(message, 'success');
        
        // Track successful playlist creation (for session counter only - main event tracked by backend)
        this.app.playlistsCreated++;
    }
    
    handlePlaylistCreationError(response, data) {
        // Handle specific error cases
        if (response.status === 403 && data.error && 
            (data.error.includes('permissions') || data.error.includes('Insufficient'))) {
            // Clear the access token and prompt for re-authentication
            this.app.accessToken = null;
            this.app.updateAuthStatus(false);
            
            // Re-check auth status since backend may have cleared session
            setTimeout(() => this.app.checkAuthStatus(), 1000);
            
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
            this.app.accessToken = null;
            this.app.updateAuthStatus(false);
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
        const exportStatus = this.app.domElements.exportStatus;
        exportStatus.innerHTML = message;
        exportStatus.className = `export-status ${type}`;
        exportStatus.style.display = 'block';
        
        // Success messages persist until accordion is closed or refreshed
    }
    
    hideExportStatus() {
        const exportStatus = this.app.domElements.exportStatus;
        exportStatus.style.display = 'none';
    }
    
    async autoLoadMoreForExport(targetCount) {
        // Keep loading more results until we have enough songs or no more results available
        while (this.app.searchResults.length < targetCount && this.app.hasMoreResults) {
            
            try {
                await this.app.loadMoreResults();
                // Update the hint after loading more results
                this.app.updateSongCountHint();
                // Brief pause to prevent overwhelming the server
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error('🎵 Failed to auto-load more results:', error);
                break;
            }
        }
        
        if (this.app.searchResults.length < targetCount && !this.app.hasMoreResults) {
            this.showExportStatus(
                `Only ${this.app.searchResults.length} songs available in total. Proceeding with all available songs.`,
                'info'
            );
            // Brief pause to show the info message
            await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
        }
    }

    generatePlaylistName() {
        if (!this.app.currentSearchType || !this.app.currentQuery) {
            return 'SongMatch';
        }
        
        if (this.app.currentSearchType === 'text') {
            // For text-to-song queries, use the query text
            const query = this.app.currentQuery.trim();
            if (query.length <= 100) {
                return query;
            } else {
                // Truncate to 97 chars + "..."
                return query.substring(0, 97) + '...';
            }
        } else if (this.app.currentSearchType === 'song' && this.app.currentQuerySong) {
            // For song-to-song queries, use "{song name} vibes"
            const songName = this.app.currentQuerySong.song || '';
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
        return 'SongMatch';
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
            if (!this.app.accessToken) {
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

    updateExportFormDisplay() {
        const songCountField = document.getElementById('song-count-field');
        const manualSelectionInfo = document.getElementById('manual-selection-info');
        const selectedSongsCount = document.getElementById('selected-songs-count');
        
        if (this.app.isManualSelectionMode) {
            // Hide number input, show selection info
            songCountField.style.display = 'none';
            manualSelectionInfo.style.display = 'flex';
            
            // Update selected count
            const count = this.app.selectedSongs.size;
            selectedSongsCount.textContent = `${count} song${count === 1 ? '' : 's'} selected`;
        } else {
            // Show number input, hide selection info
            songCountField.style.display = 'flex';
            manualSelectionInfo.style.display = 'none';
        }
    }
}