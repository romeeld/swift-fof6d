import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _m4_kernel(q):
    """
    Dimensionless M4 (cubic spline) kernel: returns W(r, h) × h³ at q = r/h.
    Compact support on [0, 1]; normalised so that ∫ W d³r = 1.
    """
    norm = 8.0 / np.pi
    W    = np.zeros_like(q, dtype=np.float64)
    m1   = q <= 0.5
    m2   = (~m1) & (q <= 1.0)
    W[m1] = norm * (1.0 - 6.0 * q[m1]**2 * (1.0 - q[m1]))
    W[m2] = norm * 2.0 * (1.0 - q[m2])**3
    return W


def _sph_velocity_dispersion(vel, dists, idxs):
    """
    SPH M4-kernel-weighted local 3-D velocity dispersion per particle.

    h_i is the distance to the last neighbour in `dists` (the n_sph-th
    nearest neighbour), giving a kernel that adapts to local density.

    Parameters
    ----------
    vel   : (N, 3)  particle velocities
    dists : (N, k)  neighbour distances from cKDTree.query (includes self)
    idxs  : (N, k)  neighbour indices   from cKDTree.query (includes self)

    Returns
    -------
    sigma : (N,)
        σ_i = sqrt( Σ_j W_ij |v_j − v̄_i|² / Σ_j W_ij )
    """
    h      = dists[:, -1]                            # (N,) smoothing lengths
    h_safe = np.where(h > 0.0, h, 1.0)              # avoid divide-by-zero

    q      = dists / h_safe[:, None]                 # (N, k)
    W      = _m4_kernel(q) / h_safe[:, None]**3      # (N, k) full kernel
    W[h == 0.0] = 0.0                                # zero degenerate rows

    W_sum      = W.sum(axis=1)                       # (N,)
    W_sum_safe = np.where(W_sum > 0.0, W_sum, 1.0)

    v_nbr = vel[idxs]                                # (N, k, 3)

    # kernel-weighted mean velocity at each particle
    v_bar = (W[:, :, None] * v_nbr).sum(axis=1) \
            / W_sum_safe[:, None]                    # (N, 3)

    dv    = v_nbr - v_bar[:, None, :]               # (N, k, 3)
    dv_sq = (dv * dv).sum(axis=-1)                  # (N, k)

    sigma2 = (W * dv_sq).sum(axis=1) / W_sum_safe   # (N,)
    sigma2[(h == 0.0) | (W_sum == 0.0)] = 0.0       # clean up edge cases
    return np.sqrt(sigma2)


def fof_6d(
    pos,
    vel,
    ids,
    f_pos,
    f_vel,
    box_size       = None,
    n_sph          = 32,
    min_group_size = 2,
):
    """
    6D Friends-of-Friends group finder with physically motivated linking lengths.

    Linking criteria
    ----------------
    Position : |Δr_ij| < b_pos  =  f_pos × (V/N)^(1/3)
    Velocity : |Δv_ij| < f_vel  ×  (σ_i + σ_j) / 2
               σ_i = SPH M4-kernel-weighted local 3-D velocity dispersion
               of particle i, estimated over its n_sph nearest neighbours
               with h_i set to the distance to the n_sph-th neighbour.

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    vel : array_like, shape (N, 3)
        Particle velocities.
    ids : array_like, shape (N,)
        Unique particle identifiers (any dtype).
    f_pos : float
        Position linking fraction.  Typical cosmological value: 0.2.
    f_vel : float
        Velocity linking multiple applied to (σ_i + σ_j) / 2.
        Typical value: 1.0 – 2.0.
    box_size : float or (3,) array_like, optional
        Enables periodic (toroidal) boundary conditions in position space.
        A scalar is broadcast to all three axes.  Positions outside [0, L)
        are wrapped automatically.
    n_sph : int, optional
        Neighbour count for the SPH kernel estimate.  Default: 32.
        Automatically capped at N − 1 for small datasets.
    min_group_size : int, optional
        Groups with fewer members receive group_id = −1.  Default: 2.

    Returns
    -------
    result : ndarray, shape (N, 2)
        Columns: [id, group_id].
        group_id ≥ 0  →  group numbered in descending-size order.
        group_id = −1 →  particle not in any qualifying group.

    Notes
    -----
    Memory scales as O(N × n_sph) for the SPH step.
    KD-tree build and pair query both run in O(N log N).
    For open boundaries, V is estimated from the axis-aligned bounding box.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    ids = np.asarray(ids)
    N   = pos.shape[0]

    if N == 0:
        return np.empty((0, 2), dtype=np.int64)

    # ── 1. Mean interparticle spacing → position linking length ──────────
    if box_size is not None:
        box_size = np.asarray(box_size, dtype=np.float64)
        if box_size.ndim == 0:
            box_size = np.full(3, float(box_size))
        volume = float(np.prod(box_size))
        pos    = pos % box_size                      # wrap into [0, L)
    else:
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        volume = float(np.prod(np.maximum(hi - lo, 1e-300)))

    b_pos = f_pos * (volume / N) ** (1.0 / 3.0)

    # ── 2. Position KD-tree (periodic-aware if box_size given) ───────────
    tree = cKDTree(pos, boxsize=box_size if box_size is not None else None)

    # ── 3. SPH-kernel local velocity dispersions ─────────────────────────
    k           = min(n_sph + 1, N)       # +1: cKDTree.query includes self
    dists, idxs = tree.query(pos, k=k)    # (N, k) — periodic distances if needed
    h           = dists[:, -1]            # Smoothing lengths
    sigma       = _sph_velocity_dispersion(vel, dists, idxs)

    # ── 4. Candidate pairs in position space: |Δr| < b_pos ───────────────
    pairs = tree.query_pairs(b_pos, output_type="ndarray")   # (M, 2)

    # ── 5. Velocity filter: |Δv| < f_vel × (σ_i + σ_j) / 2 ─────────────
    if len(pairs) > 0:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        dv           = vel[i_idx] - vel[j_idx]
        dv2          = (dv * dv).sum(axis=1)
        b_v          = f_vel * 0.5 * (sigma[i_idx] + sigma[j_idx])
        pairs        = pairs[dv2 < b_v**2]

    # ── 6. Connected components ──────────────────────────────────────────
    if len(pairs) > 0:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        rows  = np.concatenate([i_idx, j_idx])
        cols  = np.concatenate([j_idx, i_idx])
        data  = np.ones(len(rows), dtype=np.float32)
        graph = csr_matrix((data, (rows, cols)), shape=(N, N))
        _, labels = connected_components(graph, directed=False,
                                          return_labels=True)
    else:
        labels = np.arange(N, dtype=np.int32)

    # ── 7. Renumber by descending size; discard small groups ─────────────
    unique_lbls, counts = np.unique(labels, return_counts=True)
    order       = np.argsort(-counts)
    unique_lbls = unique_lbls[order]
    counts      = counts[order]

    lbl_to_gid = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
    gid = 0
    for lbl, cnt in zip(unique_lbls, counts):
        if cnt >= min_group_size:
            lbl_to_gid[lbl] = gid
            gid += 1

    group_ids = lbl_to_gid[labels]
    out = np.empty(N, dtype=[("id",               ids.dtype),
                              ("group_id",         np.int64),
                              ("smoothing_length", np.float64)])
    out["id"]               = ids
    out["group_id"]         = group_ids
    out["smoothing_length"] = h
    return out
