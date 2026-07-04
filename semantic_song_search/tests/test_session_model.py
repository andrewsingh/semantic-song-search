"""Unit tests for the playlist-builder session model (stdlib unittest, no pytest).

Run from the package dir:
    cd semantic_song_search && python -m unittest tests.test_session_model -v
or:
    cd semantic_song_search && python tests/test_session_model.py
"""
import math
import os
import sys
import unittest

import numpy as np

# Make the package modules importable regardless of the runner's cwd.
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

import constants  # noqa: E402
import session_model  # noqa: E402
from search import MusicSearchEngine, SearchConfig  # noqa: E402

SONG_ASPECTS = constants.SONG_EMBEDDING_TYPES
NON_GENRE = session_model.NON_GENRE_ARTIST_ASPECTS


def u(vec):
    v = np.asarray(vec, dtype=float)
    return v / np.linalg.norm(v)


def default_hp(**over):
    SD = constants.SESSION_DEFAULTS
    hp = {"kappa": SD["kappa"], "tau": SD["tau"], "k_pool": SD["k_pool"], "n": SD["n"],
          "max_per_artist": SD["max_per_artist"],
          "release_date_decay_constant": SD["release_date_decay_constant"]}
    hp.update(over)
    return hp


def make_song(track_id, name, artist, release_date="2020-06-15", streams_total=1e6, streams_daily=1e3):
    return {
        "track_id": track_id, "id": track_id, "name": name,
        "artists": [{"name": artist}], "uri": f"spotify:track:{track_id}",
        "album": {"name": f"{name} album", "release_date": release_date,
                  "release_date_precision": "day", "images": [{"url": "http://x/img.jpg"}]},
        "streams": {"streams_total": streams_total, "streams_daily": streams_daily},
    }


def build_engine(songs, song_embeddings, artist_embeddings, artist_metadata, genre_store):
    """Construct a MusicSearchEngine with synthetic data but the REAL methods."""
    eng = object.__new__(MusicSearchEngine)
    eng.songs = songs
    eng.embedding_lookups = song_embeddings          # {aspect: {track_id: vec}}
    eng.embedding_lookup = song_embeddings.get("genres", {})
    eng.artist_data = {"embeddings": artist_embeddings, "metadata": artist_metadata,
                       "genre_store": genre_store}
    eng.artist_embedding_lookup = {}
    eng.tags_lookup = {}
    eng.genres_lookup = {}
    eng.track_streams = {
        s["track_id"]: {
            "S_total": s["streams"]["streams_total"] / (s["streams"]["streams_total"] + 1e7),
            "S_daily": s["streams"]["streams_daily"] / (s["streams"]["streams_daily"] + 1e4),
        } for s in songs
    }
    eng.config = SearchConfig(a0_song_sim=0.55, a1_artist_sim=0.30, a2_total_streams=0.05,
                              a3_daily_streams=0.05, a4_release_date=0.05,
                              b0_genres=0.3, b1_vocal_style=0.15, b2_production_sound_design=0.15,
                              b3_lyrical_meaning=0.1, b4_mood_atmosphere=0.2, b5_tags=0.1)
    return eng


def standard_fixture():
    """A small library: 3 artists, distinct embeddings, enough to score against."""
    e = {"x": u([1, 0, 0, 0]), "y": u([0, 1, 0, 0]), "z": u([0, 0, 1, 0]),
         "xy": u([1, 1, 0, 0]), "yz": u([0, 1, 1, 0])}

    songs = [
        make_song("t0", "Seed", "Artist A", "2021-01-01"),
        make_song("t1", "Near", "Artist A", "2021-03-01"),
        make_song("t2", "Other", "Artist B", "2010-01-01"),
        make_song("t3", "Far", "Artist B", "1995-01-01"),
        make_song("t4", "Mid", "Artist C", "2018-01-01"),
        make_song("t5", "AlsoA", "Artist A", "2021-06-01"),
    ]

    # song embeddings per aspect; t0 & t1 share genres, differ on mood.
    def per_song(gen, mood):
        return {"genres": gen, "vocal_style": e["x"], "production_sound_design": e["x"],
                "lyrical_meaning": e["x"], "mood_atmosphere": mood, "tags": e["x"]}
    rows = {
        "t0": per_song(e["x"], e["x"]),
        "t1": per_song(e["x"], e["y"]),   # same genres, orthogonal mood
        "t2": per_song(e["y"], e["y"]),
        "t3": per_song(e["z"], e["z"]),
        "t4": per_song(e["xy"], e["yz"]),
        "t5": per_song(e["x"], e["xy"]),
    }
    song_embeddings = {asp: {tid: rows[tid][asp] for tid in rows} for asp in SONG_ASPECTS}

    artists = ["Artist A", "Artist B", "Artist C"]
    avecs = {"Artist A": e["x"], "Artist B": e["y"], "Artist C": e["z"]}
    artist_embeddings = {asp: {"artists": artists, "embeddings": [avecs[a] for a in artists]}
                         for asp in NON_GENRE}
    artist_metadata = {
        "Artist A": {"genres": [{"key": "genre: Pop", "prominence": 10}]},
        "Artist B": {"genres": [{"key": "genre: Rock", "prominence": 10}]},
        "Artist C": {"genres": [{"key": "genre: Pop", "prominence": 8},
                                {"key": "genre: Rock", "prominence": 4}]},
    }
    genre_store = {"genre: Pop": u([1, 0, 0, 0]), "genre: Rock": u([0, 1, 0, 0])}

    eng = build_engine(songs, song_embeddings, artist_embeddings, artist_metadata, genre_store)
    return eng, session_model.PlaylistSessionScorer(eng)


class SessionWeightTests(unittest.TestCase):
    def test_n1_weights_equal_preset(self):
        eng, scorer = standard_fixture()
        state = scorer.build_session_state([{"index": 0, "track_id": "t0", "weight": 3.0,
                                             "song": eng.songs[0], "artist": "Artist A"}], default_hp())
        self.assertAlmostEqual(state["lambda_n"], 0.0)
        preset = constants.SESSION_DEFAULTS["song_weights"]
        for asp in SONG_ASPECTS:
            self.assertAlmostEqual(state["session_song_weights"][asp], preset[asp], places=6,
                                   msg=f"aspect {asp} should equal preset at n=1")

    def test_coherence_differentiates_shared_vs_divergent_aspect(self):
        eng, scorer = standard_fixture()
        resolved = [
            {"index": 0, "track_id": "t0", "weight": 3.0, "song": eng.songs[0], "artist": "Artist A"},
            {"index": 1, "track_id": "t1", "weight": 1.0, "song": eng.songs[1], "artist": "Artist A"},
        ]
        state = scorer.build_session_state(resolved, default_hp())
        self.assertAlmostEqual(state["coherence"]["genres"], 1.0, places=6)      # identical
        self.assertLess(state["coherence"]["mood_atmosphere"], 0.95)             # divergent
        self.assertGreater(state["coherence"]["genres"], state["coherence"]["mood_atmosphere"])
        self.assertAlmostEqual(state["lambda_n"], 1.0 / 4.0)                      # (2-1)/(2-1+3)

    def test_missing_embedding_zeroes_and_renormalizes_aspect(self):
        eng, scorer = standard_fixture()
        del eng.embedding_lookups["tags"]["t0"]  # t0 loses its tags embedding
        state = scorer.build_session_state([{"index": 0, "track_id": "t0", "weight": 1.0,
                                             "song": eng.songs[0], "artist": "Artist A"}], default_hp())
        self.assertEqual(state["usable_counts"]["tags"], 0)
        self.assertEqual(state["session_song_weights"]["tags"], 0.0)
        self.assertAlmostEqual(sum(state["session_song_weights"].values()), 1.0, places=6)


class ArtistAndGenreTests(unittest.TestCase):
    def test_pooled_genre_self_normalization(self):
        eng, scorer = standard_fixture()
        scorer.defaults = dict(scorer.defaults)
        scorer.defaults["artist_weights"] = {"genres": 1.0, "vocal_style": 0, "production_sound_design": 0,
                                             "lyrical_themes": 0, "mood_atmosphere": 0, "cultural_context_scene": 0}
        state = scorer.build_session_state([{"index": 0, "track_id": "t0", "weight": 2.0,
                                             "song": eng.songs[0], "artist": "Artist A"}], default_hp())
        # Candidate with the same (proportional) genre bag → exactly 1.0.
        self.assertAlmostEqual(scorer._artist_similarity(state, "Artist A"), 1.0, places=6)
        # A different-genre artist → within [0, 1).
        other = scorer._artist_similarity(state, "Artist B")
        self.assertGreaterEqual(other, 0.0)
        self.assertLess(other, 1.0)

    def test_same_artist_p95_substitution(self):
        eng, scorer = standard_fixture()
        resolved = [{"index": 0, "track_id": "t0", "weight": 3.0, "song": eng.songs[0], "artist": "Artist A"}]
        state = scorer.build_session_state(resolved, default_hp())
        scored = scorer.score_all(state, {"t0"}, default_hp())
        by_id = {c["track_id"]: c for c in scored}
        # Recompute expected p95 over unique candidate-artist raw sims.
        artists = {c["artist"] for c in scored}
        raw = {a: scorer._artist_similarity(state, a) for a in artists}
        p95 = float(np.percentile([v for v in raw.values() if not np.isnan(v)], 95))
        # t5 is another Artist A track (anchor artist) → should carry the p95 value, not its raw sim.
        self.assertIn("t5", by_id)
        self.assertTrue(by_id["t5"]["components"]["is_same_artist"])
        self.assertAlmostEqual(by_id["t5"]["components"]["S_artist"], p95, places=6)
        # t2 (Artist B, not an anchor) keeps its own raw artist sim.
        self.assertFalse(by_id["t2"]["components"]["is_same_artist"])
        self.assertAlmostEqual(by_id["t2"]["components"]["S_artist"], raw["Artist B"], places=6)


class ScoringTests(unittest.TestCase):
    def test_release_date_inclusion_and_exclusion(self):
        eng, scorer = standard_fixture()
        # t4 has a valid date → component included; strip a candidate's date → excluded.
        eng.songs[4]["album"]["release_date"] = None
        resolved = [{"index": 0, "track_id": "t0", "weight": 1.0, "song": eng.songs[0], "artist": "Artist A"}]
        state = scorer.build_session_state(resolved, default_hp())
        scored = {c["track_id"]: c for c in scorer.score_all(state, {"t0"}, default_hp())}
        self.assertTrue(scored["t2"]["components"]["use_release_date_component"])   # has a date
        self.assertFalse(scored["t4"]["components"]["use_release_date_component"])  # date stripped

    def test_combine_matches_hand_computed(self):
        eng, scorer = standard_fixture()
        tl = constants.SESSION_DEFAULTS["top_level_weights"]
        comps = {"S_song": 0.8, "S_artist": 0.6, "S_streams_total": 0.4,
                 "S_streams_daily": 0.2, "S_release_date": 0.9}
        # With release date: all five weights renormalized.
        w = np.array([tl["a0_song_sim"], tl["a1_artist_sim"], tl["a2_total_streams"],
                      tl["a3_daily_streams"], tl["a4_release_date"]])
        vals = np.array([0.8, 0.6, 0.4, 0.2, 0.9])
        expected = float(np.dot(w / w.sum(), vals))
        self.assertAlmostEqual(scorer._combine(comps, True), expected, places=6)
        # Without release date: four weights renormalized (a4 dropped).
        w4 = w[:4]
        expected4 = float(np.dot(w4 / w4.sum(), vals[:4]))
        self.assertAlmostEqual(scorer._combine(comps, False), expected4, places=6)


class SamplerTests(unittest.TestCase):
    def _scored(self, spec):
        # spec: list of (index, artist, score)
        return [{"index": i, "artist": a, "final_score": s, "song": {}, "track_id": f"t{i}",
                 "components": {}} for (i, a, s) in spec]

    def test_determinism(self):
        eng, scorer = standard_fixture()
        scored = self._scored([(i, f"Ar{i}", 1.0 - i * 0.01) for i in range(20)])
        a = scorer.sample_options(scored, 5, np.random.default_rng(42), 0.08, 50, 1)
        b = scorer.sample_options(scored, 5, np.random.default_rng(42), 0.08, 50, 1)
        self.assertEqual([c["index"] for c in a], [c["index"] for c in b])

    def test_dedup_max_per_artist(self):
        eng, scorer = standard_fixture()
        scored = self._scored([(0, "A", 0.9), (1, "A", 0.89), (2, "A", 0.88), (3, "B", 0.5), (4, "C", 0.4)])
        picks = scorer.sample_options(scored, 3, np.random.default_rng(1), 0.08, 50, 1)
        artists = [c["artist"] for c in picks]
        self.assertEqual(len(picks), 3)
        self.assertEqual(len(set(artists)), 3)  # one each of A, B, C

    def test_backfill_beyond_pool(self):
        eng, scorer = standard_fixture()
        # top-2 pool is all artist A; B and C live beyond the pool.
        scored = self._scored([(0, "A", 0.9), (1, "A", 0.89), (2, "B", 0.5), (3, "C", 0.4)])
        picks = scorer.sample_options(scored, 3, np.random.default_rng(1), 0.08, k_pool=2, max_per_artist=1)
        self.assertEqual(len(picks), 3)
        self.assertEqual({c["artist"] for c in picks}, {"A", "B", "C"})

    def test_unlimited_when_max_none(self):
        eng, scorer = standard_fixture()
        scored = self._scored([(0, "A", 0.9), (1, "A", 0.8), (2, "A", 0.7)])
        picks = scorer.sample_options(scored, 3, np.random.default_rng(1), 0.08, 50, None)
        self.assertEqual(len(picks), 3)  # all three A's allowed


class PipelineTests(unittest.TestCase):
    def test_all_unknown_raises(self):
        eng, scorer = standard_fixture()
        with self.assertRaises(ValueError):
            scorer.suggest([("NOPE", 1.0)], set(), default_hp(), np.random.default_rng(0))

    def test_suggest_excludes_selected_and_is_deterministic(self):
        eng, scorer = standard_fixture()
        r1 = scorer.suggest([("t0", 3.0)], set(), default_hp(), np.random.default_rng(5))
        r2 = scorer.suggest([("t0", 3.0)], set(), default_hp(), np.random.default_rng(5))
        ids1 = [s["track_id"] for s in r1["suggestions"]]
        self.assertEqual(ids1, [s["track_id"] for s in r2["suggestions"]])
        self.assertNotIn("t0", ids1)
        self.assertEqual(r1["playlist"][0]["weight"], 3.0)

    def test_engine_config_not_mutated(self):
        eng, scorer = standard_fixture()
        before = eng.config.to_dict()
        scorer.suggest([("t0", 3.0), ("t2", 1.0)], set(), default_hp(), np.random.default_rng(0))
        self.assertEqual(eng.config.to_dict(), before)


class RefactorParityTests(unittest.TestCase):
    def test_build_song_result_shape(self):
        eng, scorer = standard_fixture()
        scored = eng.build_song_result(eng.songs[0], 0)  # unscored
        self.assertIsNone(scored["final_score"])
        self.assertEqual(scored["track_id"], "t0")
        self.assertEqual(scored["spotify_id"], "t0")
        withscore = eng.build_song_result(eng.songs[0], 0, song_similarity=0.5,
                                          artist_similarity=0.6, final_score=0.7, components={"a": 1})
        self.assertEqual(withscore["final_score"], 0.7)
        self.assertEqual(withscore["components"], {"a": 1})


class EndpointValidationTests(unittest.TestCase):
    """Exercise the /api/playlist/suggest validation branches with a stub scorer (no engine load)."""

    @classmethod
    def setUpClass(cls):
        import app as app_module
        cls.app_module = app_module

        class StubScorer:
            def suggest(self, playlist, exclude, hp, rng):
                if any(tid == "ALLUNKNOWN" for tid, _ in playlist):
                    raise ValueError("No known tracks in playlist.")
                return {"playlist": [], "suggestions": [], "session_diagnostics": {},
                        "unknown_track_ids": []}

        app_module.get_session_scorer = lambda: StubScorer()
        app_module.log_playlist_event = lambda record: None  # avoid file writes in tests
        cls.client = app_module.app.test_client()

    def post(self, payload):
        return self.client.post("/api/playlist/suggest", json=payload)

    def test_empty_playlist_400(self):
        self.assertEqual(self.post({"playlist": []}).status_code, 400)

    def test_missing_playlist_400(self):
        self.assertEqual(self.post({}).status_code, 400)

    def test_nonpositive_weight_400(self):
        self.assertEqual(self.post({"playlist": [{"track_id": "t0", "weight": 0}]}).status_code, 400)
        self.assertEqual(self.post({"playlist": [{"track_id": "t0", "weight": -1}]}).status_code, 400)

    def test_valueerror_maps_to_400(self):
        self.assertEqual(self.post({"playlist": [{"track_id": "ALLUNKNOWN", "weight": 1}]}).status_code, 400)

    def test_valid_request_200(self):
        r = self.post({"playlist": [{"track_id": "t0", "weight": 1}]})
        self.assertEqual(r.status_code, 200)
        self.assertIn("slate_id", r.get_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
