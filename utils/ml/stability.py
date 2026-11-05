"""Population Stability Index (PSI), shared by monitoring and the promotion gate.

PSI thresholds (industry standard):
    < 0.10      no significant shift
    0.10–0.25   minor shift, investigate
    >= 0.25     major shift, retrain
"""

import numpy as np

N_BINS = 10


def bin_scores(scores, n_bins=N_BINS):
    """Return a count-per-bin array over [0, 1) with n_bins equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(scores, bins=bins)
    return counts


def psi(reference_scores, current_scores, n_bins=N_BINS):
    """Population Stability Index between two score arrays."""
    ref_counts = bin_scores(reference_scores, n_bins).astype(float)
    cur_counts = bin_scores(current_scores, n_bins).astype(float)

    # avoid zero divisions — replace zeros with a small epsilon
    eps = 0.0001
    ref_pct = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), eps)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
