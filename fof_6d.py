import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _m4_kernel(q):
    """
    Dimensionless M4 (cubic spline) kernel: returns W(r, h) × h³ at q = r/h.
    Compact support on [0, 1]; normalised so that ∫ W d³r = 1.
    Output dtype matches input dtype.
    """
    norm  = 8.0 / np.pi
    W     = np.zeros_like(q)          # dtype follows q; no forced float64
    m1    = q <= 0.5
    m2    = (~m1) & (q <= 1.0)
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
    vel   : (N, 3)  full particle velocity array (any float dtype)
    dists : (C, k)  neighbour distances for a chunk of C particles.
                    dtype sets the working precision throughout; pass
                    float32 to keep all intermediate arrays in float32.
                    Indices in `idxs` still refer to the full N-particle
                    array, so C ≤ N.
    idxs  : (C, k)  neighbour indices from cKDTree.query (includes self)

    Returns
    -------
    sigma : (C,) array, dtype matches dists
        σ_i = sqrt( Σ_j W_ij |v_j − v̄_i|² / Σ_j W_ij )
    """
    fp     = dists.dtype               # working precision (float32 or float64)
    one    = fp.type(1.0)              # typed 1 — prevents silent float64 upcast

    h      = dists[:, -1]             # (C,) smoothing lengths
    h_safe = np.where(h > 0.0, h, one)

    q      = dists / h_safe[:, None]  # (C, k)
    W      = _m4_kernel(q) / h_safe[:, None]**3
    W[h == 0.0] = 0.0

    W_sum      = W.sum(axis=1)        # (C,)
    W_sum_safe = np.where(W_sum > 0.0, W_sum, one)

    # vel[idxs] fancy-indexes into the full array; cast to working precision
    v_nbr = vel[idxs].astype(fp, copy=False)          # (C, k, 3)

    v_bar = (W[:, :, None] * v_nbr).sum(axis=1) \
            / W_sum_safe[:, None]                      # (C, 3)

    dv    = v_nbr - v_bar[:, None, :]                 # (C, k, 3)
    dv_sq = (dv * dv).sum(axis=-1)                    # (C, k)

    sigma2 = (W * dv_sq).sum(axis=1) / W_sum_safe     # (C,)
    sigma2[(h == 0.0) | (W_sum == 0.0)] = 0.0
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
    sph_chunk      = 1_000_000,
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
        Particle positions.  Converted to float64 internally (cKDTree
        requirement); the caller's array is not modified.
    vel : array_like, shape (N, 3)
        Particle velocities.  Converted to float32 internally; the
        caller's array is not modified.
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
    sph_chunk : int, optional
        Particles processed per iteration in the SPH neighbour query.
        Caps peak memory during the SPH step to approximately:
            sph_chunk × n_sph × (4 + 4 + 4 + 4 + 12) bytes
            ≈ sph_chunk × n_sph × 28 bytes
        (distance, index, q, W, and v_nbr arrays dominate).
        Default: 1_000_000  — gives ~1–2 GB peak per chunk at n_sph = 32.
        Reduce for tighter memory budgets; increase to reduce loop overhead
        on machines with ample RAM.

    Returns
    -------
    out : structured ndarray, shape (N,)
        Fields:
            "id"               — original particle ID, input dtype preserved
            "group_id"         — int64; ≥ 0 = group index in descending-size
                                 order; −1 = not in any qualifying group
            "smoothing_length" — float32 SPH kernel smoothing length h_i,
                                 in the same units as pos

    Notes
    -----
    Positions remain float64 throughout (cKDTree requirement).
    Velocities and all SPH intermediate arrays use float32, halving their
    memory footprint versus a pure float64 implementation.
    The SPH tree query is issued in chunks of sph_chunk particles so that
    peak memory is O(sph_chunk × n_sph) rather than O(N × n_sph).
    For open boundaries V is estimated from the axis-aligned bounding box.
    """
    pos = np.asarray(pos, dtype=np.float64)   # float64: cKDTree requirement
    vel = np.asarray(vel, dtype=np.float32)   # float32: halves velocity memory
    ids = np.asarray(ids)
    N   = pos.shape[0]

    if N == 0:
        return np.empty(0, dtype=[("id",               ids.dtype),
                                   ("group_id",         np.int64),
                                   ("smoothing_length", np.float32)])

    # ── 1. Mean interparticle spacing → position linking length ──────────
    if box_size is not None:
        box_size = np.asarray(box_size, dtype=np.float64)
        if box_size.ndim == 0:
            box_size = np.full(3, float(box_size))
        volume = float(np.prod(box_size))
        pos    = pos % box_size
    else:
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        volume = float(np.prod(np.maximum(hi - lo, 1e-300)))

    b_pos = f_pos * (volume / N) ** (1.0 / 3.0)

    # ── 2. Position KD-tree (periodic-aware if box_size given) ───────────
    tree = cKDTree(pos, boxsize=box_size if box_size is not None else None)

    # ── 3. SPH-kernel local velocity dispersions (chunked, float32) ──────
    #
    # tree.query returns float64 distances; we cast to float32 immediately
    # within each iteration so the large (chunk, k) arrays never exist in
    # float64.  sigma and h are accumulated into pre-allocated float32 arrays.
    #
    k          = min(n_sph + 1, N)
    chunk_size = min(sph_chunk, N)

    h     = np.empty(N, dtype=np.float32)
    sigma = np.empty(N, dtype=np.float32)

    for start in range(0, N, chunk_size):
        end      = min(start + chunk_size, N)
        d_c, i_c = tree.query(pos[start:end], k=k)   # float64 from cKDTree
        d_c      = d_c.astype(np.float32)             # → float32 immediately

        h[start:end]     = d_c[:, -1]
        sigma[start:end] = _sph_velocity_dispersion(vel, d_c, i_c)
        # d_c and i_c are released here; only h and sigma are retained

    # ── 4. Candidate pairs in position space: |Δr| < b_pos ───────────────
    pairs = tree.query_pairs(b_pos, output_type="ndarray")   # (M, 2)

    # ── 5. Velocity filter: |Δv| < f_vel × (σ_i + σ_j) / 2 ─────────────
    if len(pairs) > 0:
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        dv           = vel[i_idx] - vel[j_idx]                       # float32
        dv2          = (dv * dv).sum(axis=1)                          # float32
        b_v          = np.float32(f_vel * 0.5) * (sigma[i_idx]
                                                   + sigma[j_idx])   # float32
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
                              ("smoothing_length", np.float32)])
    out["id"]               = ids
    out["group_id"]         = group_ids
    out["smoothing_length"] = h
    return out

