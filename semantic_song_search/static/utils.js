// Shared utilities for Semantic Song Search App

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format time from milliseconds to MM:SS
function formatTime(ms) {
    if (!ms || ms < 0) return '0:00';
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// API Helper Class
class ApiHelper {
    constructor() {
        this.defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        };
    }

    async request(endpoint, options = {}) {
        const config = { ...this.defaultOptions, ...options };
        const response = await fetch(endpoint, config);
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }
        
        return response.json();
    }

    async get(endpoint) {
        return this.request(endpoint);
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
}

// Analytics Helper Class
class AnalyticsHelper {
    constructor() {
        this.sessionStartTime = Date.now();
        this.searchCount = 0;
        this.songsPlayed = 0;
        this.playlistsCreated = 0;
        this.hasPersonalizationHistory = false;
        this.isAuthenticated = false;
    }

    getCommonTrackingProperties() {
        return {
            'timestamp': new Date().toISOString(),
            'session_time': Date.now() - this.sessionStartTime,
            'search_count': this.searchCount,
            'songs_played': this.songsPlayed,
            'playlists_created': this.playlistsCreated,
            'is_authenticated': this.isAuthenticated,
            'has_personalization_history': this.hasPersonalizationHistory
        };
    }

    trackEvent(eventName, customProperties = {}) {
        if (typeof mixpanel !== 'undefined') {
            mixpanel.track(eventName, {
                ...this.getCommonTrackingProperties(),
                ...customProperties
            });
        }
    }

    trackPageLoad() {
        if (typeof mixpanel !== 'undefined') {
            // Get timezone and language information
            const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            const language = navigator.language || navigator.userLanguage;
            const languages = navigator.languages || [language];
            
            this.trackEvent('Page Loaded', {
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

    incrementSearchCount() {
        this.searchCount++;
    }

    incrementSongsPlayed() {
        this.songsPlayed++;
    }

    incrementPlaylistsCreated() {
        this.playlistsCreated++;
    }

    updateAuthStatus(isAuthenticated) {
        this.isAuthenticated = isAuthenticated;
    }

    updatePersonalizationHistory(hasHistory) {
        this.hasPersonalizationHistory = hasHistory;
    }
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        escapeHtml,
        formatTime,
        ApiHelper,
        AnalyticsHelper
    };
}