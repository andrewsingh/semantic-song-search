// Semantic Song Search App JavaScript

class SemanticSearchApp {
    constructor() {
        // Initialize utilities and helpers
        this.api = new ApiHelper();
        this.analytics = new AnalyticsHelper();
        
        // Search state
        this.search_k = 48; // number of results to return
        this.currentQuery = null;
        this.currentQuerySong = null;
        this.currentSearchType = 'text'; // Initialize to match HTML default
        this.currentSuggestions = []; // Store current suggestions for Enter key handling
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
        this.baseSearchResults = []; // Store pre-artist-filtered results (after other filters like Top Artists)
        
        // Manual song selection
        this.isManualSelectionMode = false;
        this.selectedSongs = new Set(); // Set of song indices that are selected

        // Artist filter state (performance-optimized with hashmaps)
        this.artistFilterState = {
            isDropdownOpen: false,
            selectedArtists: new Set(), // Currently selected artists
            originalSelectedArtists: new Set(), // Original state to track changes
            artistTrackCounts: new Map(), // artist -> track count (O(1) lookups)
            artistToTracks: new Map(), // artist -> Set of track indices (O(1) lookups)
            isActive: false, // Whether artist filter is currently applied
            filteredResults: null, // Cached filtered results to avoid re-filtering
            lastFilteredWith: null // Set of artists used for last filtering (cache validation)
        };

        // Personalization controls state
        this.currentLambdaVal = 0.5;
        this.currentFamiliarityMin = 0.0;
        this.currentFamiliarityMax = 1.0;
        this.hasPersonalizationHistory = false;
        
        // No-history weights tracking (new 10-weight system)
        this.currentNoHistoryWeights = {
            // Top-level weights (a_i)
            a0_song_sim: 0.6,
            a1_artist_sim: 0.3,
            a2_total_streams: 0.05,
            a3_daily_streams: 0.05,
            a4_release_date: 0.0,

            // Song descriptor weights (b_i)
            b0_genres: 0.3,
            b1_vocal_style: 0.15,
            b2_production_sound_design: 0.15,
            b3_lyrical_meaning: 0.1,
            b4_mood_atmosphere: 0.2,
            b5_tags: 0.1
        };
        
        this.activeNoHistoryWeights = { ...this.currentNoHistoryWeights };
        
        // Cache frequently accessed DOM elements
        this.domElements = {
            topArtistsFilter: document.getElementById('top-artists-filter'),
            artistFilterDropdownBtn: document.getElementById('artist-filter-dropdown-btn'),
            artistFilterDropdown: document.getElementById('artist-filter-dropdown'),
            artistFilterList: document.querySelector('#artist-filter-dropdown .dropdown-filter-list'),
            artistFilterApplyBtn: document.getElementById('artist-filter-apply-btn'),
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
            resultsContainer: document.getElementById('results-container'),
            resultsCount: document.getElementById('results-count'),
            resultsHeader: document.getElementById('results-header'),
            playBtn: document.getElementById('play-btn'),
            prevBtn: document.getElementById('prev-btn'),
            nextBtn: document.getElementById('next-btn'),
            lambdaSlider: document.getElementById('lambda-slider'),
            lambdaValue: document.getElementById('lambda-value'),
            loadMoreBtn: document.getElementById('load-more-btn'),
            noHistoryWeightsSection: document.getElementById('no-history-weights-section'),
            noHistoryRerunBtn: document.getElementById('no-history-rerun-btn'),
            noHistoryResetBtn: document.getElementById('no-history-reset-btn'),
            // Top-level weight inputs (a_i)
            nhA0SongSim: document.getElementById('nh_a0_song_sim'),
            nhA1ArtistSim: document.getElementById('nh_a1_artist_sim'),
            nhA2TotalStreams: document.getElementById('nh_a2_total_streams'),
            nhA3DailyStreams: document.getElementById('nh_a3_daily_streams'),
            nhA4ReleaseDate: document.getElementById('nh_a4_release_date'),
            // Song descriptor weight inputs (b_i)
            nhB0Genres: document.getElementById('nh_b0_genres'),
            nhB1VocalStyle: document.getElementById('nh_b1_vocal_style'),
            nhB2ProductionSoundDesign: document.getElementById('nh_b2_production_sound_design'),
            nhB3LyricalMeaning: document.getElementById('nh_b3_lyrical_meaning'),
            nhB4MoodAtmosphere: document.getElementById('nh_b4_mood_atmosphere'),
            nhB5Tags: document.getElementById('nh_b5_tags')
        };
        
        // Store the original defaults for reset functionality
        this.defaultNoHistoryWeights = { ...this.currentNoHistoryWeights };
        
        // Initialize Spotify Player
        this.player = new SpotifyPlayer(this);

        // Initialize Playlist Export Manager
        this.playlistExport = new PlaylistExportManager(this, this.api, this.analytics);

        // Initialize new managers
        this.personalizationManager = new PersonalizationManager(this, this.api, this.analytics);
        this.resultsUIManager = new ResultsUIManager(this, this.api, this.analytics);
        this.searchManager = new SearchManager(this, this.api, this.analytics);

        this.init();
    }
    
    init() {
        this.personalizationManager.initializeWeightInputs();
        this.bindEventListeners();
        this.checkAuthStatus();

        // Ensure currentSearchType is synced with the initial HTML state
        this.currentSearchType = this.searchManager.getSearchType();


        // Track initial page load
        this.analytics.trackPageLoad();
    }
    
    bindEventListeners() {
        this.bindSearchEventListeners();
        this.bindAuthEventListeners();
        this.bindPlayerEventListeners();
        this.bindExportEventListeners();
        this.bindFilterEventListeners();
        this.bindPersonalizationEventListeners();
        this.bindNoHistoryWeightsEventListeners();
        this.bindSessionEventListeners();
    }
    
    bindSearchEventListeners() {
        // Search type change - segmented control
        const searchTypeRadios = document.querySelectorAll('input[name="search-type"]');
        if (searchTypeRadios.length > 0) {
            searchTypeRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        this.searchManager.handleSearchTypeChange(e.target.value);
                    }
                });
            });
        } else {
            console.warn('Search type radio buttons not found in DOM');
        }
                
        // Search input
        const searchInput = this.domElements.searchInput;
        searchInput.addEventListener('input', (e) => {
            this.searchManager.handleSearchInput(e.target.value);
        });
        
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchManager.handleEnterKey();
            }
        });
        
        // Load more button
        this.domElements.loadMoreBtn.addEventListener('click', () => {
            this.searchManager.loadMoreResults();
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
            this.playlistExport.toggleExportAccordion();
        });

        // Export button
        this.domElements.exportBtn.addEventListener('click', () => {
            this.playlistExport.exportToPlaylist();
        });

        // Manual selection toggle
        this.domElements.manualSelectionToggle.addEventListener('change', (e) => {
            this.resultsUIManager.toggleManualSelection(e.target.checked);
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
            
            // Reset artist filter state when Top Artists filter changes (available artists change)
            if (this.searchResults.length > 0) {
                this.artistFilterState.isActive = false;
                this.artistFilterState.selectedArtists.clear();
                this.artistFilterState.originalSelectedArtists.clear();
                this.artistFilterState.filteredResults = null;
                this.artistFilterState.lastFilteredWith = null;
            }

            // If we have existing search results, filter them client-side instead of re-running the search
            if (this.searchResults.length > 0) {
                this.searchManager.applyClientSideFilter();

                // Rebuild artist dropdown from the new base dataset
                // This ensures dropdown shows artists that are available in the base dataset
                this.resultsUIManager.buildArtistDataMaps(this.baseSearchResults, false);
                this.resultsUIManager.populateArtistFilterDropdown();
            }
        });

        // Artist filter dropdown button
        if (this.domElements.artistFilterDropdownBtn) {
            this.domElements.artistFilterDropdownBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleArtistFilterDropdown();
            });
        }

        // Artist filter Apply button
        if (this.domElements.artistFilterApplyBtn) {
            this.domElements.artistFilterApplyBtn.addEventListener('click', () => {
                this.searchManager.applyArtistFilterAndUpdate();
                this.toggleArtistFilterDropdown(); // Close dropdown after applying

                // Track artist filter usage
                this.analytics.trackEvent('Artist Filter Applied', {
                    'selected_artists_count': this.artistFilterState.selectedArtists.size,
                    'total_artists_count': this.artistFilterState.artistTrackCounts.size,
                    'filtered_results_count': this.searchResults.length,
                    'original_results_count': this.originalSearchResults.length
                });
            });
        }

        // Close artist filter dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#artist-filter-option') && this.artistFilterState.isDropdownOpen) {
                this.toggleArtistFilterDropdown();
            }
        });

        // Handle artist checkbox changes (event delegation for performance)
        if (this.domElements.artistFilterList) {
            this.domElements.artistFilterList.addEventListener('change', (e) => {
                if (e.target.classList.contains('dropdown-checkbox')) {
                    // Handle master "Select All" checkbox
                    if (e.target.dataset.action === 'select-all') {
                        if (e.target.checked) {
                            this.resultsUIManager.selectAllArtists();
                        } else {
                            this.resultsUIManager.unselectAllArtists();
                        }
                    } else {
                        // Handle individual artist checkboxes
                        const artistName = e.target.dataset.artist;
                        if (artistName) {
                            this.resultsUIManager.handleArtistCheckboxChange(artistName, e.target.checked);
                        }
                    }
                }
            });
        }
    }
    
    bindPersonalizationEventListeners() {
        // Personalization controls
        const lambdaSlider = this.domElements.lambdaSlider;
        const familiarityMinSlider = this.domElements.familiarityMin;
        const familiarityMaxSlider = this.domElements.familiarityMax;
        const rerunSearchBtn = this.domElements.rerunSearchBtn;
        
        if (lambdaSlider) {
            lambdaSlider.addEventListener('input', (e) => {
                this.personalizationManager.handleLambdaChange(parseFloat(e.target.value));
            });
        }
        
        if (familiarityMinSlider) {
            familiarityMinSlider.addEventListener('input', (e) => {
                this.personalizationManager.handleFamiliarityMinChange(parseFloat(e.target.value));
            });
        }
        
        if (familiarityMaxSlider) {
            familiarityMaxSlider.addEventListener('input', (e) => {
                this.personalizationManager.handleFamiliarityMaxChange(parseFloat(e.target.value));
            });
        }
        
        if (rerunSearchBtn) {
            rerunSearchBtn.addEventListener('click', () => {
                this.personalizationManager.rerunSearchWithNewParameters();
            });
        }
    }
    
    
    bindNoHistoryWeightsEventListeners() {
        // No-history weights input changes
        const weightInputs = [
            // Top-level weights (a_i)
            this.domElements.nhA0SongSim,
            this.domElements.nhA1ArtistSim,
            this.domElements.nhA2TotalStreams,
            this.domElements.nhA3DailyStreams,
            this.domElements.nhA4ReleaseDate,
            // Song descriptor weights (b_i)
            this.domElements.nhB0Genres,
            this.domElements.nhB1VocalStyle,
            this.domElements.nhB2ProductionSoundDesign,
            this.domElements.nhB3LyricalMeaning,
            this.domElements.nhB4MoodAtmosphere,
            this.domElements.nhB5Tags
        ];

        weightInputs.forEach(input => {
            if (input) {
                input.addEventListener('input', (e) => {
                    this.personalizationManager.handleNoHistoryWeightChange(e.target.id, e.target.value);
                });
            }
        });

        // No-history rerun button
        const noHistoryRerunBtn = this.domElements.noHistoryRerunBtn;
        if (noHistoryRerunBtn) {
            noHistoryRerunBtn.addEventListener('click', () => {
                this.personalizationManager.rerunSearchWithNoHistoryWeights();
            });
        }

        // No-history reset button
        const noHistoryResetBtn = this.domElements.noHistoryResetBtn;
        if (noHistoryResetBtn) {
            noHistoryResetBtn.addEventListener('click', () => {
                this.personalizationManager.resetNoHistoryWeightsToDefaults();
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
            "Welcome to SongMatch! 🎵\n\n" +
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

    toggleArtistFilterDropdown() {
        /**
         * Toggle the artist filter dropdown open/closed
         */
        const artistFilter = this.artistFilterState;
        const dropdown = this.domElements.artistFilterDropdown;
        const arrow = this.domElements.artistFilterDropdownBtn?.querySelector('.dropdown-filter-arrow');

        if (!dropdown) return;

        artistFilter.isDropdownOpen = !artistFilter.isDropdownOpen;

        if (artistFilter.isDropdownOpen) {
            dropdown.style.display = 'block';
            if (arrow) arrow.textContent = '▲';
        } else {
            dropdown.style.display = 'none';
            if (arrow) arrow.textContent = '▼';
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