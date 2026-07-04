// Playlist Builder — testbed UI for the session model v2 (set-as-evidence).
// The playlist is the selected set S; each mutation POSTs the full set to
// /api/playlist/suggest and re-renders the hydrated playlist + N suggestions.
// Reuses SpotifyPlayer, ApiHelper, AnalyticsHelper, the card CSS, and the
// /api/search_suggestions + /api/create_playlist endpoints.

class PlaylistBuilder {
    constructor() {
        const SD = window.SESSION_DEFAULTS || {};
        this.SD = SD;

        this.api = new ApiHelper();
        this.analytics = new AnalyticsHelper();
        // Reused for the exact search-result card markup (header, tags, accordion, footer).
        this.resultsUI = new ResultsUIManager(this, this.api, this.analytics);

        // Player-shim fields SpotifyPlayer reads on its `app`.
        this.accessToken = null;
        this.isAuthenticated = false;
        this.searchResults = [];       // transient single-track preview context
        this.searchResultsId = 'pb';
        this.player = new SpotifyPlayer(this);

        // Session state.
        this.playlistEntries = [];     // [{track_id, weight}] — request source of truth
        this.playlistSongs = [];       // hydrated song objects from the server
        this.suggestions = [];         // hydrated suggestion objects
        this.shownExclude = new Set(); // suggestion ids excluded across rerolls
        this.prevSlateId = null;
        this.diagnostics = null;

        this.hyperparams = {
            kappa: SD.kappa,
            tau: SD.tau,
            n: SD.n,
            max_per_artist: SD.max_per_artist,
        };

        this.currentSuggestions = []; // fuzzy search dropdown results
        this.suggestBusy = false;

        this.cacheDom();
        this.bindEvents();
        this.initControls();
        this.checkAuthStatus();
        this.renderAll();

        if (this.player.initSpotifyPlayer) {
            // Lazily initialized on first playback; safe to attempt now if token exists.
        }
    }

    cacheDom() {
        this.el = {
            searchInput: document.getElementById('pb-search-input'),
            suggestionsDropdown: document.getElementById('pb-suggestions'),
            playlistGrid: document.getElementById('pb-playlist-grid'),
            playlistCount: document.getElementById('pb-playlist-count'),
            suggestionsGrid: document.getElementById('pb-suggestions-grid'),
            rerollBtn: document.getElementById('pb-reroll-btn'),
            devPanel: document.getElementById('pb-dev-panel'),
            devToggle: document.getElementById('pb-dev-toggle'),
            devDiagnostics: document.getElementById('pb-dev-diagnostics'),
            exportBtn: document.getElementById('pb-export-btn'),
            playlistName: document.getElementById('pb-playlist-name'),
            exportStatus: document.getElementById('pb-export-status'),
            ctlKappa: document.getElementById('pb-ctl-kappa'),
            ctlTau: document.getElementById('pb-ctl-tau'),
            ctlN: document.getElementById('pb-ctl-n'),
            ctlMaxArtist: document.getElementById('pb-ctl-maxartist'),
            ctlApply: document.getElementById('pb-ctl-apply'),
        };
    }

    bindEvents() {
        let debounce;
        this.el.searchInput.addEventListener('input', (e) => {
            clearTimeout(debounce);
            const q = e.target.value.trim();
            if (q.length < 2) { this.hideDropdown(); return; }
            debounce = setTimeout(() => this.fetchFuzzy(q), 180);
        });
        document.addEventListener('click', (e) => {
            if (!this.el.searchInput.contains(e.target) && !this.el.suggestionsDropdown.contains(e.target)) {
                this.hideDropdown();
            }
        });

        this.el.rerollBtn.addEventListener('click', () => this.reroll());
        this.el.exportBtn.addEventListener('click', () => this.exportPlaylist());
        this.el.devToggle.addEventListener('click', () => this.toggleDevPanel());
        this.el.ctlApply.addEventListener('click', () => this.applyControls());

        // Player controls (reused SpotifyPlayer).
        document.getElementById('play-btn').addEventListener('click', () => this.player.togglePlayback());
        document.getElementById('next-btn').addEventListener('click', () => this.player.nextTrack());
        document.getElementById('prev-btn').addEventListener('click', () => this.player.previousTrack());
        document.getElementById('progress-bar').addEventListener('click', (e) => this.player.seekToPosition(e));
        document.getElementById('login-btn').addEventListener('click', () => { window.location.href = '/login'; });
        document.getElementById('logout-btn').addEventListener('click', () => { window.location.href = '/logout'; });
    }

    initControls() {
        this.el.ctlKappa.value = this.hyperparams.kappa;
        this.el.ctlTau.value = this.hyperparams.tau;
        this.el.ctlN.value = this.hyperparams.n;
        this.el.ctlMaxArtist.value = this.hyperparams.max_per_artist ?? 0;
    }

    applyControls() {
        this.hyperparams.kappa = parseFloat(this.el.ctlKappa.value);
        this.hyperparams.tau = parseFloat(this.el.ctlTau.value);
        this.hyperparams.n = parseInt(this.el.ctlN.value, 10);
        this.hyperparams.max_per_artist = parseInt(this.el.ctlMaxArtist.value, 10) || 0;
        if (this.playlistEntries.length > 0) this.suggest('controls');
    }

    // ---- auth (player shim) ---------------------------------------------

    async checkAuthStatus() {
        try {
            const data = await this.api.get('/api/token');
            this.accessToken = data.access_token;
            this.updateAuthStatus(true);
        } catch (e) {
            this.updateAuthStatus(false);
        }
    }

    updateAuthStatus(isAuthenticated) {
        this.isAuthenticated = isAuthenticated;
        const indicator = document.getElementById('auth-indicator');
        const text = document.getElementById('auth-text');
        const loginBtn = document.getElementById('login-btn');
        const logoutBtn = document.getElementById('logout-btn');
        if (isAuthenticated) {
            indicator.textContent = '●';
            indicator.className = 'auth-indicator connected';
            text.textContent = 'Connected to Spotify';
            loginBtn.style.display = 'none';
            logoutBtn.style.display = 'inline-block';
        } else {
            indicator.textContent = '○';
            indicator.className = 'auth-indicator';
            text.textContent = 'Not connected to Spotify';
            loginBtn.style.display = 'inline-block';
            logoutBtn.style.display = 'none';
        }
    }

    showError(message) {
        this.setExportStatus(message, 'error');
    }

    // ---- fuzzy search dropdown ------------------------------------------

    async fetchFuzzy(query) {
        try {
            const results = await this.api.get(`/api/search_suggestions?query=${encodeURIComponent(query)}`);
            this.currentSuggestions = results || [];
            this.renderDropdown();
        } catch (e) {
            this.hideDropdown();
        }
    }

    renderDropdown() {
        const box = this.el.suggestionsDropdown;
        if (!this.currentSuggestions.length) { this.hideDropdown(); return; }
        box.innerHTML = this.currentSuggestions.slice(0, 12).map((s, i) => `
            <div class="suggestion-item" data-idx="${i}">
                <img class="suggestion-cover" src="${escapeHtml(s.cover_url || '')}" alt="">
                <div class="suggestion-details">
                    <div class="suggestion-title">${escapeHtml(s.song)}</div>
                    <div class="suggestion-artist">${escapeHtml(s.artist)}</div>
                </div>
            </div>`).join('');
        box.querySelectorAll('.suggestion-item').forEach((node) => {
            node.addEventListener('click', () => {
                const s = this.currentSuggestions[parseInt(node.dataset.idx, 10)];
                this.selectFromSearch(s);
            });
        });
        box.style.display = 'block';
    }

    hideDropdown() { this.el.suggestionsDropdown.style.display = 'none'; }

    selectFromSearch(s) {
        this.el.searchInput.value = '';
        this.hideDropdown();
        const isSeed = this.playlistEntries.length === 0;
        this.addTrack(s.spotify_id, isSeed ? this.SD.seed_anchor_weight : this.SD.normal_weight,
                      isSeed ? 'seed' : 'manual_add');
    }

    // ---- playlist mutations (each triggers a suggest) -------------------

    entryFor(trackId) { return this.playlistEntries.find((e) => e.track_id === trackId); }

    addTrack(trackId, weight, trigger) {
        if (!trackId) return;
        const existing = this.entryFor(trackId);
        if (existing) { existing.weight = weight; }
        else { this.playlistEntries.push({ track_id: trackId, weight }); }
        this.suggest(trigger, [trackId]);
    }

    removeTrack(trackId) {
        this.playlistEntries = this.playlistEntries.filter((e) => e.track_id !== trackId);
        if (this.playlistEntries.length === 0) {
            this.playlistSongs = [];
            this.suggestions = [];
            this.diagnostics = null;
            this.shownExclude.clear();
            this.prevSlateId = null;
            this.renderAll();
            return;
        }
        this.suggest('remove', [trackId]);
    }

    toggleAnchor(trackId) {
        const entry = this.entryFor(trackId);
        if (!entry) return;
        entry.weight = (entry.weight >= this.SD.seed_anchor_weight)
            ? this.SD.normal_weight : this.SD.seed_anchor_weight;
        this.suggest('anchor', [trackId]);
    }

    reroll() {
        if (!this.playlistEntries.length) return;
        this.suggest('reroll');
    }

    // ---- the core call --------------------------------------------------

    async suggest(trigger, changedTrackIds = null) {
        if (this.suggestBusy) return;
        this.suggestBusy = true;
        this.el.rerollBtn.disabled = true;

        // Reset the reroll-exclusion window on any non-reroll change.
        if (trigger !== 'reroll') this.shownExclude.clear();

        try {
            const body = {
                playlist: this.playlistEntries.map((e) => ({ track_id: e.track_id, weight: e.weight })),
                n: this.hyperparams.n,
                exclude: Array.from(this.shownExclude),
                hyperparams: {
                    kappa: this.hyperparams.kappa,
                    tau: this.hyperparams.tau,
                    n: this.hyperparams.n,
                    max_per_artist: this.hyperparams.max_per_artist,
                },
                trigger,
                changed_track_ids: changedTrackIds,
                prev_slate_id: this.prevSlateId,
            };
            const res = await this.api.post('/api/playlist/suggest', body);
            this.playlistSongs = res.playlist || [];
            this.suggestions = res.suggestions || [];
            this.diagnostics = res.session_diagnostics || null;
            this.prevSlateId = res.slate_id || null;
            this.suggestions.forEach((s) => this.shownExclude.add(s.track_id));
            this.renderAll();
        } catch (e) {
            const msg = (e && e.message) ? e.message : 'Suggestion request failed.';
            this.setExportStatus(msg, 'error');
        } finally {
            this.suggestBusy = false;
            this.el.rerollBtn.disabled = this.playlistEntries.length === 0;
        }
    }

    // ---- rendering ------------------------------------------------------

    renderAll() {
        this.renderPlaylist();
        this.renderSuggestions();
        this.renderDiagnostics();
    }

    // Card reuses the exact search-result markup (header, tags, accordion, footer)
    // from ResultsUIManager; events are bound locally per zone.
    makeCard(song, { zone, rank, weight }) {
        const card = document.createElement('div');
        card.className = 'song-card pb-card';
        const isAnchor = zone === 'playlist' && weight >= this.SD.seed_anchor_weight;
        if (isAnchor) card.classList.add('pb-anchor');

        // Suggestion cards show the rank + scoring footer; playlist cards don't.
        const opts = (zone === 'suggestion') ? { rank, similarity: song.final_score } : {};
        card.innerHTML = this.resultsUI.createSongCardHTML(song, opts);

        // Zone-specific action buttons overlaid top-right.
        const actions = document.createElement('div');
        actions.className = 'pb-card-actions';
        if (zone === 'playlist') {
            actions.innerHTML =
                `<button class="pb-icon pb-anchor-btn" title="Anchor (weight this song more)">${isAnchor ? '★' : '☆'}</button>
                 <button class="pb-icon pb-remove-btn" title="Remove from playlist">✕</button>`;
        } else {
            actions.innerHTML = `<button class="pb-icon pb-add-btn" title="Add to playlist">＋</button>`;
        }
        card.insertBefore(actions, card.firstChild);

        // Accordion toggle (reuse the manager's handler).
        const accordion = card.querySelector('.accordion-toggle');
        if (accordion) {
            accordion.addEventListener('click', (e) => {
                e.stopPropagation(); this.resultsUI.toggleAccordion(accordion);
            });
        }

        // Preview on card-body click (excluding the action buttons + accordion).
        const header = card.querySelector('.card-header');
        if (header) header.addEventListener('click', () => this.playPreview(song));

        if (zone === 'playlist') {
            actions.querySelector('.pb-anchor-btn').addEventListener('click', (e) => {
                e.stopPropagation(); this.toggleAnchor(song.track_id);
            });
            actions.querySelector('.pb-remove-btn').addEventListener('click', (e) => {
                e.stopPropagation(); this.removeTrack(song.track_id);
            });
        } else {
            actions.querySelector('.pb-add-btn').addEventListener('click', (e) => {
                e.stopPropagation(); this.addTrack(song.track_id, this.SD.normal_weight, 'add');
            });
        }
        return card;
    }

    renderPlaylist() {
        const grid = this.el.playlistGrid;
        this.el.playlistCount.textContent = `(${this.playlistSongs.length})`;
        if (!this.playlistSongs.length) {
            grid.innerHTML = '<div class="pb-empty">Search for a song above to start your playlist.</div>';
            return;
        }
        grid.innerHTML = '';
        this.playlistSongs.forEach((song) => {
            grid.appendChild(this.makeCard(song, { zone: 'playlist', weight: song.weight }));
        });
    }

    renderSuggestions() {
        const grid = this.el.suggestionsGrid;
        if (!this.suggestions.length) {
            grid.innerHTML = '<div class="pb-empty">Suggestions appear once you add a seed song.</div>';
            return;
        }
        grid.innerHTML = '';
        this.suggestions.forEach((song, i) => {
            grid.appendChild(this.makeCard(song, { zone: 'suggestion', rank: i + 1 }));
        });
    }

    renderDiagnostics() {
        const box = this.el.devDiagnostics;
        if (!this.diagnostics) {
            box.innerHTML = '<div class="pb-empty">Diagnostics appear after the first suggestion.</div>';
            return;
        }
        const d = this.diagnostics;
        const rows = Object.entries(d.song_aspects).map(([aspect, a]) => {
            const rpct = Math.round(a.R * 100);
            return `<tr>
                <td class="pb-aspect">${aspect}</td>
                <td class="pb-barcell"><div class="pb-bar" style="width:${rpct}%"></div><span>${a.R.toFixed(3)}</span></td>
                <td class="pb-num">${a.session_weight.toFixed(3)}</td>
                <td class="pb-num pb-muted">${a.preset_weight}</td>
                <td class="pb-num pb-muted">${a.usable_count}/${d.n}</td>
            </tr>`;
        }).join('');
        box.innerHTML = `
            <div class="pb-diag-meta">n=${d.n} · λ<sub>n</sub>=${d.lambda_n} · κ=${d.kappa} · τ=${d.tau}
                · N=${d.n_suggestions} · max/artist=${d.max_per_artist ?? '∞'} · era anchor=${d.era_anchor || '—'}</div>
            <table class="pb-diag-table">
                <thead><tr>
                    <th>aspect</th>
                    <th>coherence R<sub>i</sub></th>
                    <th>session weight</th>
                    <th>preset</th>
                    <th title="Playlist songs that have an embedding for this aspect, out of n. Aspects with 0 usable get weight 0.">usable</th>
                </tr></thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    // ---- player + export ------------------------------------------------

    playPreview(song) {
        // V0: single-track preview, no auto-advance.
        this.searchResults = [song];
        this.player.isAutoPlayEnabled = false;
        if (this.player.currentSongIndex !== undefined) this.player.currentSongIndex = 0;
        this.player.playSong(song);
    }

    toggleDevPanel() {
        const open = this.el.devPanel.style.display !== 'none';
        this.el.devPanel.style.display = open ? 'none' : 'block';
        this.el.devToggle.setAttribute('aria-expanded', String(!open));
    }

    setExportStatus(message, type) {
        this.el.exportStatus.textContent = message;
        this.el.exportStatus.className = `pb-export-status pb-status-${type}`;
    }

    async exportPlaylist() {
        if (!this.isAuthenticated || !this.accessToken) {
            if (confirm('Connect your Spotify account to export playlists. Login now?')) {
                window.location.href = '/login';
            }
            return;
        }
        if (!this.playlistSongs.length) {
            this.setExportStatus('Add some songs first.', 'error'); return;
        }
        const name = (this.el.playlistName.value || '').trim() || 'Songpath session';
        const ids = this.playlistSongs.map((s) => s.spotify_id).filter(Boolean);
        this.setExportStatus('Creating playlist…', 'info');
        try {
            const res = await this.api.post('/api/create_playlist', {
                playlist_name: name,
                song_count: ids.length,
                song_spotify_ids: ids,
                search_context: { source: 'playlist_builder' },
            });
            if (res && res.success) {
                this.el.exportStatus.innerHTML =
                    `Created <a href="${res.playlist_url}" target="_blank" rel="noopener">${escapeHtml(res.playlist_name)}</a> (${res.songs_added} tracks).`;
                this.el.exportStatus.className = 'pb-export-status pb-status-success';
            } else {
                this.setExportStatus((res && res.error) || 'Export failed.', 'error');
            }
        } catch (e) {
            this.setExportStatus((e && e.message) || 'Export failed.', 'error');
        }
    }
}

window.PlaylistBuilder = PlaylistBuilder;
