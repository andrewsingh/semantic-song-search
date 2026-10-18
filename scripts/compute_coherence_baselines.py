#!/usr/bin/env python3
"""Compute per-aspect coherence baselines for the session model (Refinement 1).

For each aspect i, the baseline R_bar_i^(n) is the *expected* resultant length of a random
n-song subset. Rather than Monte-Carlo it, we use the exact closed form: for n unit vectors
drawn at random from the library's aspect-i embeddings,

    E[R^2] = 1/n + (1 - 1/n) * rho_i^2 ,     rho_i = || mean(unit embeddings) ||

so  R_bar_i^(n) = sqrt( 1/n + (1 - 1/n) * rho_i^2 ).

rho_i is the aspect's *global concentration* (the library's own mean resultant length on that
aspect) — computed once over the entire library. The normalized coherence used by the scorer is
then  R_hat_i = R_i / R_bar_i^(n).

We also run a Monte-Carlo check on the song aspects to confirm the closed form matches empirics
(the only approximation being E[R] ~= sqrt(E[R^2]), a small Jensen gap we report).

Output: <library-dir>/library_v4_coherence_baselines.json
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

SONG_ASPECTS = ["genres", "vocal_style", "production_sound_design",
                "lyrical_meaning", "mood_atmosphere", "tags"]
# Artist non-genre aspects (genres are handled separately via prominence cross-similarity).
ARTIST_ASPECTS = ["vocal_style", "production_sound_design", "lyrical_themes",
                  "mood_atmosphere", "cultural_context_scene"]
MC_NS = [2, 3, 5, 10, 20]


def unit_rows(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def concentration(unit_matrix):
    """rho = || mean unit vector || = global mean resultant length for the aspect."""
    return float(np.linalg.norm(unit_matrix.mean(axis=0)))


def rbar_closed_form(rho, n):
    return float(np.sqrt(1.0 / n + (1.0 - 1.0 / n) * rho * rho))


def mc_rbar(unit_matrix, n, samples, rng):
    total = len(unit_matrix)
    vals = np.empty(samples)
    for s in range(samples):
        idx = rng.choice(total, size=n, replace=False)
        vals[s] = np.linalg.norm(unit_matrix[idx].sum(axis=0)) / n
    return float(vals.mean()), float(vals.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-dir", default="data/library_v4")
    ap.add_argument("--mc-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    emb_dir = os.path.join(args.library_dir, "library_v4_embeddings")
    art_dir = os.path.join(args.library_dir, "library_v4_artist_embeddings")

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "library_dir": args.library_dir,
        "formula": "R_bar(n) = sqrt(1/n + (1 - 1/n) * rho^2)",
        "usage": "R_hat_i = R_i / R_bar_i(n); rho_i = global mean resultant length of unit embeddings.",
        "song_aspects": {},
        "artist_aspects": {},
    }

    # --- song aspects: closed form + Monte-Carlo validation ---
    print("=== song aspects ===")
    for aspect in SONG_ASPECTS:
        data = np.load(os.path.join(emb_dir, f"{aspect}_embeddings.npz"), allow_pickle=True)
        unit = unit_rows(data["embeddings"])
        rho = concentration(unit)
        entry = {
            "rho": rho,
            "rho_squared": rho * rho,
            "n_vectors": int(len(unit)),
            "closed_form": {str(n): rbar_closed_form(rho, n) for n in MC_NS},
            "monte_carlo": {},
        }
        for n in MC_NS:
            mean, std = mc_rbar(unit, n, args.mc_samples, rng)
            entry["monte_carlo"][str(n)] = {"mean": mean, "std": std}
        result["song_aspects"][aspect] = entry
        maxdiff = max(abs(entry["closed_form"][str(n)] - entry["monte_carlo"][str(n)]["mean"])
                      for n in MC_NS)
        print(f"  {aspect:24s} rho={rho:.4f}  max|closed-MC|={maxdiff:.4f}")

    # --- artist non-genre aspects: closed form over the per-track artist distribution ---
    # (a session's artist centroid is over the artists of the selected *tracks*, so the null
    #  model is the artist of a random track — i.e. weight each artist by its track count.)
    print("=== artist non-genre aspects (per-track) ===")
    main_artists = np.load(os.path.join(emb_dir, "genres_embeddings.npz"),
                           allow_pickle=True)["main_artists"]
    for aspect in ARTIST_ASPECTS:
        data = np.load(os.path.join(art_dir, f"{aspect}_artist_embeddings.npz"), allow_pickle=True)
        names = [str(a) for a in data["artists"]]
        vecs = unit_rows(data["embeddings"])
        lut = {names[i]: vecs[i] for i in range(len(names))}
        per_track = np.array([lut[str(a)] for a in main_artists if str(a) in lut])
        rho = concentration(per_track)
        result["artist_aspects"][aspect] = {
            "rho": rho,
            "rho_squared": rho * rho,
            "n_vectors": int(len(per_track)),
            "n_unique_artists": len(names),
            "closed_form": {str(n): rbar_closed_form(rho, n) for n in MC_NS},
        }
        print(f"  {aspect:24s} rho={rho:.4f}  (per-track n={len(per_track)}, unique artists={len(names)})")

    out_path = os.path.join(args.library_dir, "library_v4_coherence_baselines.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
