# swift-fof6d

6D phase-space Friends-of-Friends (FoF) group finder for
[SWIFT](https://swift.strw.leidenuniv.nl/) cosmological simulation snapshots.

Groups are defined by linking particles that are close in both position and velocity space simultaneously, making this complementary to SWIFT's built-in 3D position-only FoF finder. Phase-space linking is particularly effective at identifying kinematically coherent structures and suppressing spurious bridging between haloes that are close in projection but dynamically distinct.

## Features
- Links particles in combined position + velocity space
- Position linking length set as a fraction of the mean interparticle spacing
- Velocity linking length set as a multiple of the local SPH velocity dispersion
- Periodic boundary conditions supported
- Runs FoF on dark matter; assigns all other particle types (gas, stars, black holes)
  to groups via nearest-DM-neighbour

## Dependencies
- numpy
- scipy
- h5py

## Usage
```bash
python swift_fof6d.py snapshot_0050.hdf5 --f-pos 0.2 --f-vel 1.5 --n-sph 32 --min-size 20
```

## Output
Default: <snapshot>_fof6d.hdf5  [can be overwritten with --out option]
│
├── Header/                         (attributes)
│   ├── SnapshotFile                path to the input SWIFT snapshot
│   ├── Scale-factor                a
│   ├── Redshift                    z
│   ├── PartTypes                   int32 array of particle types present
│   ├── N_groups                    number of groups above min_group_size
│   ├── N_grouped_DM                number of DM particles in groups
│   ├── f_pos                       position linking fraction used
│   ├── f_vel                       velocity linking multiple used
│   ├── n_sph                       SPH neighbour count used
│   └── min_group_size              minimum group size used
│
├── PartType0/                      Gas (if present)
│   ├── ParticleIDs    (N_gas,)     original SWIFT particle IDs
│   └── GroupID        (N_gas,)     group index; -1 = ungrouped
│
├── PartType1/                      Dark matter
│   ├── ParticleIDs    (N_dm,)      original SWIFT particle IDs
│   ├── GroupID        (N_dm,)      group index; -1 = ungrouped
│   └── SmoothingLength(N_dm,)      SPH kernel smoothing length h_i
│                                   [comoving, same units as snapshot positions]
│
├── PartType4/                      Stars (if present)
│   ├── ParticleIDs    (N_star,)    original SWIFT particle IDs
│   └── GroupID        (N_star,)    group index; -1 = ungrouped
│
├── PartType5/                      Black holes (if present)
│   ├── ParticleIDs    (N_bh,)      original SWIFT particle IDs
│   └── GroupID        (N_bh,)      group index; -1 = ungrouped
│
└── Groups/                         Group catalogue (DM-based)
    ├── GroupID        (G,)          group indices, descending DM size order
    ├── Size           (G,)          DM member count per group
    └── Mass           (G,)          total DM mass per group [snapshot mass units]

## Reading the output
```bash
import h5py
import numpy as np

with h5py.File("snapshot_0050_fof6d.hdf5", "r") as f:

    # Per-particle arrays
    dm_ids   = f["PartType1/ParticleIDs"][:]      # uint64
    dm_gids  = f["PartType1/GroupID"][:]           # int64; -1 = ungrouped
    dm_h     = f["PartType1/SmoothingLength"][:]   # float64, comoving Mpc
    gas_gids = f["PartType0/GroupID"][:]

    # Group catalogue
    grp_ids  = f["Groups/GroupID"][:]              # descending size order
    sizes    = f["Groups/Size"][:]
    masses   = f["Groups/Mass"][:]

    # Metadata
    z        = f["Header"].attrs["Redshift"]
    n_groups = f["Header"].attrs["N_groups"]

# Select all DM particles belonging to the largest group
largest_gid = grp_ids[0]
mask        = dm_gids == largest_gid
print(f"Largest group: {mask.sum()} DM particles")
```

## Performance Notes
- KD-tree construction and pair queries run in $O(N \log N)$
- The SPH velocity dispersion step allocates an Ndm x n-sph, e.g. for Ndm=1e8 and n-sph=32 this is 25 GB.  For very large simulations, consider reducing n-sph or running on a high-memory node
- Nearest-neighbour assignment for non-DM particle types is fast ($O(M \log N)$ per type) and requires rebuilding the DM KD-tree once per type


