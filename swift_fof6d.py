"""
swift_fof6d.py
==============
6-D phase-space Friends-of-Friends on a SWIFT cosmological HDF5 snapshot.

Command-line
------------
    python swift_fof6d.py snapshot_0050.hdf5
    python swift_fof6d.py snapshot_0050.hdf5 --f-pos 0.2 --f-vel 1.5 --min-size 20

Programmatic
------------
    from swift_fof6d import run_fof
    result, cat = run_fof("snapshot_0050.hdf5")

Output HDF5 layout
------------------
    /Header                      snapshot + FoF metadata (attributes)
    /Particles/ParticleIDs  (N,) original SWIFT particle IDs
    /Particles/GroupID      (N,) group index; -1 = ungrouped
    /Groups/GroupID         (G,) group indices, descending size order
    /Groups/Size            (G,) member count per group
    /Groups/Mass            (G,) total mass [snapshot mass units]  (if available)
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from fof_6d import fof_6d        # module from the previous step


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_PTYPE_NAMES = {0: "Gas", 1: "DarkMatter", 4: "Stars", 5: "BlackHoles"}

# CGS reference values → human-readable labels
_LEN_UNITS  = {3.085678e24: "Mpc",      3.085678e22: "kpc",   3.085678e21: "pc"}
_VEL_UNITS  = {1.000000e+5: "km/s",     1.000000e+0: "cm/s"}
_MASS_UNITS = {1.989000e43: "10¹⁰ M⊙", 1.989000e33: "M⊙"}

def _unit_label(cgs_val, table, default="internal units"):
    if cgs_val is None or np.isnan(cgs_val):
        return default
    for ref, name in table.items():
        if abs(cgs_val / ref - 1.0) < 0.05:
            return name
    return default

def _to_scalar(x):
    """
    Safely extract a Python float from an HDF5 attribute.
    h5py can return scalar attributes as 0-d or 1-element ndarrays;
    .flat[0] handles both without triggering the NumPy 1.25 deprecation.
    """
    return float(np.asarray(x).flat[0])


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot loading
# ═══════════════════════════════════════════════════════════════════════════

def load_swift_snapshot(path, part_type=1):
    """
    Read particle data from a SWIFT HDF5 snapshot.

    Works with single-file snapshots and with distributed multi-file
    snapshots accessed through SWIFT's top-level virtual HDF5 file
    (automatically created by SWIFT ≥ 0.9).

    Parameters
    ----------
    path      : str or Path
    part_type : int — 0 Gas | 1 DarkMatter (default) | 4 Stars | 5 BH

    Returns
    -------
    pos    : (N, 3) float64   comoving positions        [internal U_L]
    vel    : (N, 3) float64   peculiar velocities       [internal U_v]
    ids    : (N,)   int64     particle IDs
    box    : (3,)   float64   comoving box side lengths [same as pos]
    meta   : dict             units, cosmology, per-particle masses
    """
    with h5py.File(path, "r") as f:

        # ── Header ────────────────────────────────────────────────────────
        hdr = f["Header"].attrs

        box = np.atleast_1d(np.array(hdr["BoxSize"], dtype=np.float64)).ravel()
        if box.size == 1:
            box = np.full(3, box[0])

        a = _to_scalar(hdr["Scale-factor"])          
        z = _to_scalar(hdr["Redshift"])              

        # ── Units ─────────────────────────────────────────────────────────
        # SWIFT writes "Unit X in cgs (U_X)" attributes in /Units.
        # Fallbacks: Mpc / km s⁻¹ / 10¹⁰ M⊙  (common EAGLE/FLAMINGO setup)
        def _attr(group, key, fallback=np.nan):
            try:
                return _to_scalar(f[group].attrs[key])   # line 98
            except (KeyError, TypeError, ValueError, IndexError):
                return fallback

        UL = _attr("Units", "Unit length in cgs (U_L)",   3.085678e24)
        Uv = _attr("Units", "Unit velocity in cgs (U_v)", 1.000000e+5)
        UM = _attr("Units", "Unit mass in cgs (U_M)",     1.989000e43)

        # ── Cosmology ─────────────────────────────────────────────────────
        H0      = _attr("Cosmology", "H0 [internal units]")
        Omega_m = _attr("Cosmology", "Omega_m")
        Omega_L = _attr("Cosmology", "Omega_lambda")

        # ── Particles ─────────────────────────────────────────────────────
        pkey = f"PartType{part_type}"
        if pkey not in f:
            available = [k for k in f if k.startswith("PartType")]
            raise KeyError(f"'{pkey}' not found. Available: {available}")

        pgrp = f[pkey]
        pos  = pgrp["Coordinates"][:]   # (N, 3) comoving [U_L]
        vel  = pgrp["Velocities"][:]    # (N, 3) peculiar  [U_v]
        ids  = pgrp["ParticleIDs"][:]   # (N,)

        # Masses: per-particle array or uniform value from MassTable
        if "Masses" in pgrp:
            masses = pgrp["Masses"][:].astype(np.float64)
        else:
            mt = hdr.get("MassTable", None)
            if mt is not None:
                m0 = float(np.atleast_1d(mt)[part_type])
                masses = np.full(len(ids), m0, dtype=np.float64) if m0 > 0 else None
            else:
                masses = None

    meta = dict(
        a=a, z=z, UL=UL, Uv=Uv, UM=UM,
        H0=H0, Omega_m=Omega_m, Omega_L=Omega_L,
        part_type=part_type, masses=masses,
    )
    return pos.astype(np.float64), vel.astype(np.float64), ids, box, meta


# ═══════════════════════════════════════════════════════════════════════════
# Group catalogue
# ═══════════════════════════════════════════════════════════════════════════

def build_catalogue(dm_result, masses=None):
    """
    Summarise FoF groups from the DM structured result array.

    Parameters
    ----------
    dm_result : structured ndarray with fields ("id", "group_id")
    masses    : (N_dm,) float64 or None
    """
    gid  = dm_result["group_id"]               # already int64
    mask = gid >= 0

    if not mask.any():
        return {"group_ids": np.array([], dtype=np.int64),
                "sizes":     np.array([], dtype=np.int64)}

    unique_gids, sizes = np.unique(gid[mask], return_counts=True)
    order       = np.argsort(-sizes)
    unique_gids = unique_gids[order]
    sizes       = sizes[order]
    cat         = {"group_ids": unique_gids, "sizes": sizes}

    if masses is not None:
        max_gid                 = int(unique_gids.max()) + 1
        gid_to_bin              = np.full(max_gid, -1, dtype=np.intp)
        gid_to_bin[unique_gids] = np.arange(len(unique_gids), dtype=np.intp)
        bins                    = gid_to_bin[gid[mask]]
        cat["masses"]           = np.bincount(bins, weights=masses[mask],
                                              minlength=len(unique_gids))
    return cat

# ═══════════════════════════════════════════════════════════════════════════
# Assign other part types to same group as nearest DM particle
# ═══════════════════════════════════════════════════════════════════════════

def assign_by_nearest_dm(dm_pos, dm_group_ids, other_pos, box_size=None):
    """
    Assign a group ID to every particle in `other_pos` by inheriting the
    group ID of its nearest DM neighbour.

    A non-DM particle whose closest DM neighbour is ungrouped (group_id = -1)
    also receives group_id = -1.

    Parameters
    ----------
    dm_pos       : (N_dm, 3) float64  DM positions, already wrapped into [0, L)
    dm_group_ids : (N_dm,)   int64    per-DM-particle group IDs
    other_pos    : (M, 3)    float64  positions of the target particle type
    box_size     : (3,) float64 or None

    Returns
    -------
    group_ids : (M,) int64

    Notes
    -----
    Rebuilds the DM KD-tree internally.  For many particle types this is
    called once per type; the rebuild cost is O(N_dm log N_dm) and is
    small relative to the FoF computation.
    """
    if box_size is not None:
        other_pos = other_pos % box_size       # wrap into same domain as DM tree
    tree = cKDTree(dm_pos, boxsize=box_size)
    _, nn_idx = tree.query(other_pos, k=1)
    return dm_group_ids[nn_idx]

# ═══════════════════════════════════════════════════════════════════════════
# HDF5 output
# ═══════════════════════════════════════════════════════════════════════════

def write_output(out_path, all_results, cat, snap_path, meta, params):
    """Write per-particle-type group assignments and group catalogue to HDF5."""
    kw = dict(compression="gzip", compression_opts=4)
    with h5py.File(out_path, "w") as f:

        hdr = f.create_group("Header")
        hdr.attrs.update({
            "SnapshotFile"   : str(snap_path),
            "Scale-factor"   : meta["a"],
            "Redshift"       : meta["z"],
            "PartTypes"      : np.array(sorted(all_results.keys()),
                                        dtype=np.int32),
            "N_groups"       : len(cat["group_ids"]),
            "N_grouped_DM"   : int((all_results[1]["group_id"] >= 0).sum()),
            "f_pos"          : params["f_pos"],
            "f_vel"          : params["f_vel"],
            "n_sph"          : params["n_sph"],
            "min_group_size" : params["min_group_size"],
        })

        for ptype, result in all_results.items():
            pg = f.create_group(f"PartType{ptype}")
            pg.attrs["GroupAssignmentMethod"] = \
                "6D-FoF" if ptype == 1 else "NearestDMNeighbour"
            pg.create_dataset("ParticleIDs", data=result["id"],       **kw)
            pg.create_dataset("GroupID",     data=result["group_id"], **kw)
            if "smoothing_length" in result.dtype.names:
                ds = pg.create_dataset("SmoothingLength",
                                       data=result["smoothing_length"], **kw)
                ds.attrs["Units"] = _unit_label(meta["UL"], _LEN_UNITS)
                ds.attrs["Comoving"] = True
                ds.attrs["Description"] = (
                    "SPH kernel smoothing length; equals the distance to the "
                    "n_sph-th nearest DM neighbour in comoving coordinates."
                )

        gg = f.create_group("Groups")
        gg.create_dataset("GroupID", data=cat["group_ids"], **kw)
        gg.create_dataset("Size",    data=cat["sizes"],     **kw)
        if "masses" in cat:
            ds = gg.create_dataset("Mass", data=cat["masses"], **kw)
            ds.attrs["Units"] = _unit_label(meta["UM"], _MASS_UNITS)

# ═══════════════════════════════════════════════════════════════════════════
# Terminal summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(box, meta, cat, all_results, params, t_fof):
    dm_result = all_results[1]
    N         = len(dm_result)
    sizes     = cat["sizes"]
    n_groups  = len(sizes)

    len_unit  = _unit_label(meta["UL"], _LEN_UNITS)
    mass_unit = _unit_label(meta["UM"], _MASS_UNITS)
    ell       = (np.prod(box) / N) ** (1.0 / 3.0)
    b_pos_v   = params["f_pos"] * ell
    W         = 66

    print("═" * W)
    print("  6-D Phase-Space FoF  ·  SWIFT snapshot")
    print("═" * W)
    print(f"  Redshift          z  =  {meta['z']:.4f}")
    print(f"  Scale factor      a  =  {meta['a']:.6f}")
    for key, lbl in [("Omega_m", "Ω_m"), ("Omega_L", "Ω_Λ")]:
        v = meta[key]
        if not np.isnan(v):
            print(f"  {lbl:<17}  =  {v:.4f}")
    box_str = " × ".join(f"{b:.3f}" for b in box)
    print(f"  Box               =  {box_str}  [{len_unit}, comoving]")
    print(f"  N DM particles    =  {N:,}")
    print(f"  Mean spacing  ℓ   =  {ell:.5f}  [{len_unit}]")
    h = dm_result["smoothing_length"]
    print(f"  Smoothing length  =  "
          f"median {np.median(h):.5f},  "
          f"range [{h.min():.5f}, {h.max():.5f}]  [{len_unit}]")
    print()
    print(f"  f_pos = {params['f_pos']}  →  b_pos = {b_pos_v:.5f} [{len_unit}]")
    print(f"  f_vel = {params['f_vel']}  →  b_vel = {params['f_vel']} × σ_v  "
          f"(n_sph = {params['n_sph']})")
    print(f"  FoF wall time     =  {t_fof:.1f} s")
    print("─" * W)
    print(f"  Groups  (≥ {params['min_group_size']} members)  =  {n_groups:,}")
    if n_groups:
        n_dm_grp = int((dm_result["group_id"] >= 0).sum())
        print(f"  Largest group     =  {sizes[0]:,} DM particles")
        print(f"  Median group size =  {int(np.median(sizes)):,} DM particles")

    # ── Per-particle-type breakdown ───────────────────────────────────────
    print("─" * W)
    print(f"  {'Type':<16} {'N total':>14} {'N grouped':>12}  "
          f"{'%':>6}  Method")
    print("  " + "─" * (W - 2))
    for ptype, result in sorted(all_results.items()):
        pname  = _PTYPE_NAMES.get(ptype, f"PartType{ptype}")
        n_p    = len(result)
        n_g    = int((result["group_id"] >= 0).sum())
        method = "6D-FoF" if ptype == 1 else "nearest DM"
        print(f"  {pname:<16} {n_p:>14,} {n_g:>12,}  "
              f"{100*n_g/n_p:>5.1f}%  {method}")
    print("═" * W)

    if not n_groups:
        return

    # ── Size distribution ─────────────────────────────────────────────────
    ms     = params["min_group_size"]
    edges  = [ms, 50, 100, 200, 500, 1_000, 5_000, 10_000, np.inf]
    labels = ["< 50","50–99","100–199","200–499",
              "500–999","1k–5k","5k–10k","≥ 10k"]
    print("\n  Group size distribution (DM members):")
    for lo, hi, lbl in zip(edges[:-1], edges[1:], labels):
        n = int(np.sum((sizes >= lo) & (sizes < hi)))
        if n:
            print(f"    {lbl:>9s}  {n:7,}  {'▮' * min(n, 40)}")

    # ── Top-10 table ──────────────────────────────────────────────────────
    has_mass = "masses" in cat
    print(f"\n  Top 10 groups by DM size:")
    hdr_str = f"    {'GID':>6}  {'DM members':>12}"
    if has_mass:
        hdr_str += f"  {'Mass':>18}  [{mass_unit}]"
    print(hdr_str)
    print("    " + "─" * (len(hdr_str) - 4))
    for k in range(min(10, n_groups)):
        row = f"    {int(cat['group_ids'][k]):>6}  {int(sizes[k]):>12,}"
        if has_mass:
            row += f"  {cat['masses'][k]:>22.4e}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════

def run_fof(
    snap_path,
    f_pos          = 0.2,
    f_vel          = 1.5,
    n_sph          = 32,
    min_group_size = 20,
    out_path       = None,
    verbose        = True,
):
    """
    Load a SWIFT snapshot, run 6-D FoF on DM (PartType1), then assign
    group IDs to all other particle types present in the snapshot by
    nearest-DM-neighbour inheritance.

    Parameters
    ----------
    snap_path      : str or Path
    f_pos          : float  position linking fraction        (default 0.2)
    f_vel          : float  velocity linking multiple        (default 1.5)
    n_sph          : int    SPH neighbours for σ_v estimate  (default 32)
    min_group_size : int    minimum DM particles per group   (default 20)
    out_path       : str or Path or None
    verbose        : bool

    Returns
    -------
    all_results : dict {part_type (int): structured ndarray}
        Fields: ("id", "group_id").  Keys are all PartType numbers present
        in the snapshot.  group_id = -1 means ungrouped.
    cat : dict
        Group catalogue built from DM groups (group_ids, sizes, [masses]).
    """
    snap_path = Path(snap_path)
    out_path  = Path(out_path) if out_path else \
                snap_path.with_name(snap_path.stem + "_fof6d.hdf5")
    params = dict(f_pos=f_pos, f_vel=f_vel,
                  n_sph=n_sph, min_group_size=min_group_size)

    # ── Discover which particle types are present ─────────────────────────
    with h5py.File(snap_path, "r") as f:
        num_part = np.array(f["Header"].attrs["NumPart_Total"], dtype=np.int64)
    present_types = [i for i in range(6) if num_part[i] > 0]
    if 1 not in present_types:
        raise RuntimeError(
            "No DM particles (PartType1) found; cannot define FoF groups.")
    other_types = [t for t in present_types if t != 1]

    # ── 1. Load DM ────────────────────────────────────────────────────────
    if verbose:
        print(f"Loading DarkMatter from  {snap_path.name}  ...")
    t0 = time.perf_counter()
    dm_pos, dm_vel, dm_ids, box, meta = load_swift_snapshot(snap_path,
                                                             part_type=1)
    if verbose:
        print(f"  {len(dm_ids):,} particles loaded in "
              f"{time.perf_counter()-t0:.1f}s")

    mem_gb = len(dm_ids) * n_sph * 8 / 1e9
    if verbose and mem_gb > 4.0:
        print(f"  ⚠  SPH step will use ≈ {mem_gb:.1f} GB  "
              f"(N={len(dm_ids):,}, n_sph={n_sph})")

    # ── 2. Run 6-D FoF on DM ─────────────────────────────────────────────
    if verbose:
        print("Running 6-D FoF on DarkMatter ...")
    t1 = time.perf_counter()
    dm_result = fof_6d(                          # structured array directly
        dm_pos, dm_vel, dm_ids,
        f_pos=f_pos, f_vel=f_vel,
        box_size=box, n_sph=n_sph,
        min_group_size=min_group_size,
    )
    t_fof = time.perf_counter() - t1
    # Convert to structured array — avoids float64 promotion from uint64 IDs
    all_results = {1: dm_result}

    # ── 3. Assign group IDs to all other particle types ───────────────────
    # fof_6d wraps positions as pos % box_size internally; replicate that here
    # so both trees are built from identically wrapped coordinates.
    dm_pos_wrapped   = dm_pos % box
    dm_group_ids_arr = dm_result["group_id"]      # (N_dm,) int64 view

    for ptype in other_types:
        pname = _PTYPE_NAMES.get(ptype, f"PartType{ptype}")
        if verbose:
            print(f"Assigning group IDs to {pname} ...")
        t2 = time.perf_counter()

        o_pos, _, o_ids, _, _ = load_swift_snapshot(snap_path, part_type=ptype)
        o_gids = assign_by_nearest_dm(dm_pos_wrapped, dm_group_ids_arr,
                                      o_pos, box)

        o_result = np.empty(len(o_ids),
                            dtype=[("id", o_ids.dtype), ("group_id", np.int64)])
        o_result["id"]       = o_ids
        o_result["group_id"] = o_gids
        all_results[ptype]   = o_result

        if verbose:
            n_g = int((o_gids >= 0).sum())
            print(f"  {len(o_ids):,} particles, {n_g:,} grouped "
                  f"({100*n_g/len(o_ids):.1f}%)  "
                  f"[{time.perf_counter()-t2:.1f}s]")

    # ── 4. Group catalogue (DM-based) ─────────────────────────────────────
    cat = build_catalogue(dm_result, masses=meta["masses"])

    # ── 5. Write & report ────────────────────────────────────────────────
    write_output(out_path, all_results, cat, snap_path, meta, params)

    if verbose:
        print_summary(box, meta, cat, all_results, params, t_fof)
        print(f"\n  Saved  →  {out_path}")

    return all_results, cat

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser(
        prog="swift_fof6d",
        description="6-D phase-space FoF on DM; assigns all other types "
                    "by nearest-DM-neighbour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("snapshot")
    p.add_argument("--f-pos",    type=float, default=0.2)
    p.add_argument("--f-vel",    type=float, default=1.5)
    p.add_argument("--n-sph",    type=int,   default=32)
    p.add_argument("--min-size", type=int,   default=20)
    p.add_argument("--out",      default=None)
    a = p.parse_args()
    run_fof(
        a.snapshot,
        f_pos=a.f_pos, f_vel=a.f_vel, n_sph=a.n_sph,
        min_group_size=a.min_size, out_path=a.out,
    )

if __name__ == "__main__":
    _cli()

