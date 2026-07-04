"""Session model (set-as-evidence / latent-vibe) scorer for the playlist builder.

Implements the V0 session model from docs/session_model_v2.md on top of the
existing MusicSearchEngine. The playlist is the selected set S; per song aspect
we compute a centroid (mu_i) and coherence (R_i = resultant length), modulate the
preset aspect weights by coherence (with small-n shrinkage), score every library
candidate against the centroids, and sample N suggestions with a seeded softmax +
configurable artist dedup.

The scorer holds its own weights as plain data and never reads or writes
engine.config, and never calls engine.compute_final_score — so /api/search is
untouched and there is no shared-state race.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from . import constants, data_utils
except ImportError:  # running as a script from within the package dir
    import constants
    import data_utils

logger = logging.getLogger(__name__)

# Artist aspects excluding genres (genres are handled via the pooled-bag path).
NON_GENRE_ARTIST_ASPECTS = [a for a in constants.ARTIST_EMBEDDING_TYPES if a != "genres"]


def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    """Weighted median: smallest value whose cumulative weight reaches half the total."""
    pairs = sorted(zip(values, weights), key=lambda pw: pw[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    half = total / 2.0
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= half:
            return value
    return pairs[-1][0]


def _centroid_and_coherence(vectors: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, float]:
    """Return (unit-normalized centroid, resultant length R in [0,1]) for unit vectors."""
    matrix = np.vstack(vectors)
    w = np.asarray(weights, dtype=float)
    weighted_sum = (matrix * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(weighted_sum))
    total_w = float(w.sum())
    coherence = norm / total_w if total_w > 0 else 0.0
    centroid = weighted_sum / norm if norm > 1e-12 else weighted_sum
    return centroid, float(min(max(coherence, 0.0), 1.0))


class PlaylistSessionScorer:
    def __init__(self, engine, defaults: Optional[Dict[str, Any]] = None):
        self.engine = engine
        self.defaults = defaults if defaults is not None else constants.SESSION_DEFAULTS
        # track_id -> index into engine.songs
        self._track_index: Dict[str, int] = {}
        for idx, song in enumerate(engine.songs):
            track_id = song.get("track_id") or song.get("id")
            if track_id:
                self._track_index[track_id] = idx
        # Per-aspect artist embedding lookups (name -> unit vector), built from the
        # correct path engine.artist_data['embeddings'][aspect].
        self._artist_lookups = self._build_artist_lookups()

    # ---- data access -----------------------------------------------------

    def _build_artist_lookups(self) -> Dict[str, Dict[str, np.ndarray]]:
        lookups: Dict[str, Dict[str, np.ndarray]] = {}
        artist_data = getattr(self.engine, "artist_data", None)
        if not artist_data or "embeddings" not in artist_data:
            return lookups
        embeddings = artist_data["embeddings"]
        for aspect in NON_GENRE_ARTIST_ASPECTS:
            entry = embeddings.get(aspect)
            if not entry:
                continue
            names = entry.get("artists")
            vecs = entry.get("embeddings")
            if names is None or vecs is None:
                continue
            aspect_map: Dict[str, np.ndarray] = {}
            for name, vec in zip(names, vecs):
                vector = np.asarray(vec, dtype=float)
                norm = np.linalg.norm(vector)
                if norm > 1e-12:
                    vector = vector / norm
                aspect_map[str(name)] = vector
            lookups[aspect] = aspect_map
        return lookups

    def _artist_of(self, song: Dict) -> str:
        artists = song.get("artists") or []
        return artists[0].get("name", "") if artists else ""

    def _artist_genres(self, artist_name: str) -> List[Dict]:
        artist_data = getattr(self.engine, "artist_data", None)
        if not artist_data or "metadata" not in artist_data:
            return []
        meta = artist_data["metadata"].get(artist_name)
        return meta.get("genres", []) if meta else []

    def resolve_playlist(self, playlist: List[Tuple[str, float]]) -> Tuple[List[Dict], List[str]]:
        """Filter to known tracks. Returns (resolved rows, unknown track_ids)."""
        resolved, unknown = [], []
        for track_id, weight in playlist:
            idx = self._track_index.get(track_id)
            if idx is None:
                unknown.append(track_id)
                continue
            resolved.append({
                "index": idx,
                "track_id": track_id,
                "weight": float(weight),
                "song": self.engine.songs[idx],
                "artist": self._artist_of(self.engine.songs[idx]),
            })
        return resolved, unknown

    # ---- session state ---------------------------------------------------

    def build_session_state(self, resolved: List[Dict], hp: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-aspect centroids/coherence, session weights, artist state, era anchor."""
        n = len(resolved)
        kappa = float(hp["kappa"])
        lam = (n - 1) / (n - 1 + kappa) if (n - 1 + kappa) > 0 else 0.0
        preset_song = self.defaults["song_weights"]

        # --- song aspects: centroids, coherence, session weights ---
        centroids: Dict[str, np.ndarray] = {}
        coherence: Dict[str, float] = {}
        usable_counts: Dict[str, int] = {}
        for aspect in constants.SONG_EMBEDDING_TYPES:
            lookup = self.engine.embedding_lookups.get(aspect, {})
            vecs, weights = [], []
            for row in resolved:
                emb = lookup.get(row["track_id"])
                if emb is not None:
                    vecs.append(np.asarray(emb, dtype=float))
                    weights.append(row["weight"])
            usable_counts[aspect] = len(vecs)
            if vecs:
                centroids[aspect], coherence[aspect] = _centroid_and_coherence(vecs, weights)
            else:
                coherence[aspect] = 0.0

        effective: Dict[str, float] = {}
        for aspect in constants.SONG_EMBEDDING_TYPES:
            if usable_counts[aspect] == 0:
                effective[aspect] = 0.0
                continue
            r_tilde = (1.0 - lam) + lam * coherence[aspect]  # Phase 1: raw R (no baseline)
            effective[aspect] = preset_song.get(aspect, 0.0) * r_tilde
        total_eff = sum(effective.values())
        session_song_weights = {
            aspect: (val / total_eff if total_eff > 0 else 0.0)
            for aspect, val in effective.items()
        }

        # --- artist state: per-aspect centroids (weights left at preset in Phase 1) ---
        selected_artists = [(row["artist"], row["weight"]) for row in resolved if row["artist"]]
        artist_centroids: Dict[str, np.ndarray] = {}
        artist_coherence: Dict[str, float] = {}
        for aspect in NON_GENRE_ARTIST_ASPECTS:
            lookup = self._artist_lookups.get(aspect, {})
            vecs, weights = [], []
            for artist_name, weight in selected_artists:
                vec = lookup.get(artist_name)
                if vec is not None:
                    vecs.append(vec)
                    weights.append(weight)
            if vecs:
                artist_centroids[aspect], artist_coherence[aspect] = _centroid_and_coherence(vecs, weights)
            else:
                artist_coherence[aspect] = 0.0

        # Pooled latent genre profile G(S): merge each selected artist's (key, prominence)
        # bag scaled by the per-track weight.
        pooled_genres: Dict[str, float] = {}
        for artist_name, weight in selected_artists:
            for genre in self._artist_genres(artist_name):
                key = genre.get("key")
                if key is None:
                    continue
                pooled_genres[key] = pooled_genres.get(key, 0.0) + weight * float(genre.get("prominence", 0))
        genre_bag = [{"key": k, "prominence": p} for k, p in pooled_genres.items()]
        genre_self_sim = self.engine.compute_pairwise_weighted_sum(genre_bag, genre_bag) if genre_bag else 0.0

        # --- era anchor: weighted median release date of selected songs ---
        dates, date_weights = [], []
        for row in resolved:
            album = row["song"].get("album", {}) or {}
            parsed = data_utils.parse_release_date(
                album.get("release_date"), album.get("release_date_precision")
            )
            if parsed is not None:
                dates.append(parsed)
                date_weights.append(row["weight"])
        era_anchor = _weighted_median(dates, date_weights) if dates else None

        return {
            "n": n,
            "lambda_n": lam,
            "centroids": centroids,
            "coherence": coherence,
            "usable_counts": usable_counts,
            "session_song_weights": session_song_weights,
            "artist_centroids": artist_centroids,
            "artist_coherence": artist_coherence,
            "genre_bag": genre_bag,
            "genre_self_sim": genre_self_sim,
            "anchor_artists": {a for a, _ in selected_artists},
            "era_anchor": era_anchor,
        }

    # ---- scoring ---------------------------------------------------------

    def _artist_similarity(self, state: Dict[str, Any], candidate_artist: str) -> float:
        preset_artist = self.defaults["artist_weights"]
        num, den = 0.0, 0.0

        # Genre component: self-normalized cross-sim (product form = cosine of the
        # prominence-weighted genre vectors).
        w_genres = preset_artist.get("genres", 0.0)
        if w_genres > 0 and state["genre_bag"]:
            cand_bag = self._artist_genres(candidate_artist)
            if cand_bag:
                cross = self.engine.compute_pairwise_weighted_sum(state["genre_bag"], cand_bag)
                self_c = self.engine.compute_pairwise_weighted_sum(cand_bag, cand_bag)
                self_s = state["genre_self_sim"]
                if self_s > 0 and self_c > 0:
                    gsim = float(np.clip(cross / math.sqrt(self_s * self_c), 0.0, 1.0))
                    num += w_genres * gsim
                    den += w_genres

        # Non-genre aspects: cosine to the artist centroid.
        for aspect in NON_GENRE_ARTIST_ASPECTS:
            centroid = state["artist_centroids"].get(aspect)
            if centroid is None:
                continue
            cand_vec = self._artist_lookups.get(aspect, {}).get(candidate_artist)
            if cand_vec is None:
                continue
            weight = preset_artist.get(aspect, 0.0)
            if weight <= 0:
                continue
            num += weight * float(np.dot(centroid, cand_vec))
            den += weight

        return num / den if den > 0 else 0.0

    def _era_similarity(self, state: Dict[str, Any], song: Dict, gamma: float) -> Optional[float]:
        anchor = state["era_anchor"]
        if anchor is None:
            return None
        album = song.get("album", {}) or {}
        cand_date = data_utils.parse_release_date(
            album.get("release_date"), album.get("release_date_precision")
        )
        if cand_date is None:
            return None
        days = abs((cand_date - anchor).days)
        return float(math.exp(-days / gamma))

    def _combine(self, comps: Dict[str, float], use_release_date: bool) -> float:
        tl = self.defaults["top_level_weights"]
        keys = ["a0_song_sim", "a1_artist_sim", "a2_total_streams", "a3_daily_streams"]
        vals = [comps["S_song"], comps["S_artist"], comps["S_streams_total"], comps["S_streams_daily"]]
        if use_release_date:
            keys.append("a4_release_date")
            vals.append(comps["S_release_date"])
        weights = np.array([tl[k] for k in keys], dtype=float)
        total = weights.sum()
        if total <= 1e-8:
            return float(np.clip(comps["S_song"], 0.0, 1.0))
        score = float(np.dot(weights / total, np.array(vals, dtype=float)))
        return float(np.clip(score, 0.0, 1.0))

    def score_all(self, state: Dict[str, Any], exclude_ids: set, hp: Dict[str, Any]) -> List[Dict]:
        """Score every candidate (library minus exclude_ids). Returns list sorted by final score desc."""
        gamma = float(hp["release_date_decay_constant"])
        song_weights = state["session_song_weights"]
        centroids = state["centroids"]

        # Cache artist similarity per unique candidate artist.
        artist_sim_cache: Dict[str, float] = {}

        # First pass: raw artist sims + song sims + era, collect components.
        rows: List[Dict] = []
        for idx, song in enumerate(self.engine.songs):
            track_id = song.get("track_id") or song.get("id")
            if not track_id or track_id in exclude_ids:
                continue
            candidate_artist = self._artist_of(song)

            song_sim = self.engine.compute_song_descriptor_similarity(
                centroids, track_id, song_weights, "song"
            )
            if candidate_artist not in artist_sim_cache:
                artist_sim_cache[candidate_artist] = self._artist_similarity(state, candidate_artist)
            artist_sim_raw = artist_sim_cache[candidate_artist]

            streams = self.engine.track_streams.get(track_id, {})
            era_sim = self._era_similarity(state, song, gamma)

            rows.append({
                "index": idx,
                "song": song,
                "track_id": track_id,
                "artist": candidate_artist,
                "song_sim": float(song_sim),
                "artist_sim_raw": float(artist_sim_raw),
                "s_streams_total": float(streams.get("S_total", 0.0)),
                "s_streams_daily": float(streams.get("S_daily", 0.0)),
                "era_sim": era_sim,
            })

        # Same-artist p95 mitigation (over unique candidate-artist sims).
        unique_sims = [s for s in artist_sim_cache.values() if not np.isnan(s)]
        p95 = float(np.percentile(unique_sims, 95)) if unique_sims else 0.0
        anchor_artists = state["anchor_artists"]

        scored: List[Dict] = []
        for row in rows:
            artist_sim = p95 if row["artist"] in anchor_artists else row["artist_sim_raw"]
            use_rd = row["era_sim"] is not None
            comps = {
                "S_song": row["song_sim"],
                "S_artist": artist_sim,
                "S_streams_total": row["s_streams_total"],
                "S_streams_daily": row["s_streams_daily"],
                "S_release_date": row["era_sim"] if use_rd else 0.0,
                "is_same_artist": row["artist"] in anchor_artists,
                "use_release_date_component": use_rd,
            }
            final_score = self._combine(comps, use_rd)
            comps["final_score"] = final_score
            scored.append({
                "index": row["index"],
                "song": row["song"],
                "track_id": row["track_id"],
                "artist": row["artist"],
                "final_score": final_score,
                "components": comps,
            })

        # Deterministic base ordering (score desc, then index for stable ties).
        scored.sort(key=lambda c: (-c["final_score"], c["index"]))
        return scored

    # ---- sampling --------------------------------------------------------

    def sample_options(self, scored: List[Dict], n: int, rng: np.random.Generator,
                       tau: float, k_pool: int, max_per_artist: Optional[int]) -> List[Dict]:
        """Top-K softmax sampling without replacement + artist dedup + deterministic backfill."""
        pool = scored[:k_pool]
        chosen: List[Dict] = []
        chosen_indices: set = set()
        artist_counts: Dict[str, int] = {}

        def dedup_ok(artist: str) -> bool:
            if not max_per_artist:
                return True
            return artist_counts.get(artist, 0) < max_per_artist

        remaining = list(range(len(pool)))
        scores = np.array([c["final_score"] for c in pool], dtype=float)
        while len(chosen) < n and remaining:
            eligible = [i for i in remaining if dedup_ok(pool[i]["artist"])]
            if not eligible:
                break
            logits = scores[eligible] / max(tau, 1e-6)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            pick = int(rng.choice(eligible, p=probs))
            cand = pool[pick]
            chosen.append(cand)
            chosen_indices.add(cand["index"])
            artist_counts[cand["artist"]] = artist_counts.get(cand["artist"], 0) + 1
            remaining.remove(pick)

        # Backfill deterministically down the full ranked list if dedup starved the pool.
        if len(chosen) < n:
            for cand in scored:
                if len(chosen) >= n:
                    break
                if cand["index"] in chosen_indices or not dedup_ok(cand["artist"]):
                    continue
                chosen.append(cand)
                chosen_indices.add(cand["index"])
                artist_counts[cand["artist"]] = artist_counts.get(cand["artist"], 0) + 1

        return chosen

    # ---- orchestration ---------------------------------------------------

    def _diagnostics(self, state: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
        preset_song = self.defaults["song_weights"]
        aspects = {}
        for aspect in constants.SONG_EMBEDDING_TYPES:
            aspects[aspect] = {
                "R": round(state["coherence"].get(aspect, 0.0), 4),
                "session_weight": round(state["session_song_weights"].get(aspect, 0.0), 4),
                "preset_weight": preset_song.get(aspect, 0.0),
                "usable_count": state["usable_counts"].get(aspect, 0),
            }
        return {
            "n": state["n"],
            "lambda_n": round(state["lambda_n"], 4),
            "kappa": hp["kappa"],
            "tau": hp["tau"],
            "n_suggestions": hp["n"],
            "max_per_artist": hp["max_per_artist"],
            "song_aspects": aspects,
            "era_anchor": state["era_anchor"].isoformat() if state["era_anchor"] is not None else None,
        }

    def suggest(self, playlist: List[Tuple[str, float]], exclude_ids: set,
                hp: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """Full pipeline. Raises ValueError if no known track remains after resolution."""
        resolved, unknown = self.resolve_playlist(playlist)
        if not resolved:
            raise ValueError("No known tracks in playlist.")

        state = self.build_session_state(resolved, hp)

        selected_ids = {row["track_id"] for row in resolved}
        exclude = set(exclude_ids) | selected_ids
        scored = self.score_all(state, exclude, hp)
        options = self.sample_options(
            scored, int(hp["n"]), rng, float(hp["tau"]), int(hp["k_pool"]),
            hp["max_per_artist"],
        )

        playlist_out = [
            self.engine.build_song_result(row["song"], row["index"])
            for row in resolved
        ]
        # attach the per-track weight so the client can render anchors
        for out, row in zip(playlist_out, resolved):
            out["weight"] = row["weight"]

        suggestions = [
            self.engine.build_song_result(
                opt["song"], opt["index"],
                song_similarity=opt["components"]["S_song"],
                artist_similarity=opt["components"]["S_artist"],
                final_score=opt["final_score"],
                components=opt["components"],
            )
            for opt in options
        ]

        return {
            "playlist": playlist_out,
            "suggestions": suggestions,
            "session_diagnostics": self._diagnostics(state, hp),
            "unknown_track_ids": unknown,
        }
