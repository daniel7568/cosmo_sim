# galaxy_sim_with_hecate.py
"""
Auto-download HECATE (fallback: GLADE+) and run the galaxy-body simulation.
Save snapshots to data/galaxy_sim_hecate.npz

Requirements:
  - numpy
  - scipy (optional)
  - pandas
  - numba (optional but recommended)
  - requests
Install e.g.: pip install numpy pandas numba requests
"""

import os
import io
import sys
import math
import numpy as np
import pandas as pd
from time import perf_counter

# try Numba
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

# ----------------- CONFIG --------------------
HECATE_PAGE = "https://hecate.ia.forth.gr/"   # human page (we'll try known download endpoints)
# Fallback direct GLADE+ download (public link referenced by GLADE authors)
GLADE_PLUS_URL = "http://elysium.elte.hu/~dalyag/GLADE+.txt"

CATALOG_LOCAL = "data/galaxy_catalog_hecate.csv"  # where catalog will be saved
MAX_GALAXIES = 20             # number of galaxy centers to load (tune for speed)
N_BODIES_PER_GAL = 50
SPAWN_SIGMA_AU = 2e5
TIME_YEARS = 1e6
SAVE_INTERVAL_YEARS = 1e4
INITIAL_DT = 1.0
G = 4 * math.pi**2  # AU^3 / (yr^2 * M_sun)

# thresholds
TIDAL_RADIUS_FACTOR = 5.0
MERGE_DIST_AU = 1e7
SPLIT_FRAC = 0.4

# ---------- small helpers ----------
AU_PER_PC = 206264.806
AU_PER_MPC = AU_PER_PC * 1e6

def try_download_hecate(dest_path):
    """
    Attempt to download a CSV from HECATE. The website provides CSV, VOTable, FITS, IPAC.
    We try a few guesses, and then fail gracefully.
    """
    import requests, time
    tried = []
    # common guesses (the site lists CSV download option; exact endpoint may vary)
    urls_to_try = [
        "https://hecate.ia.forth.gr/catalog.csv",
        "https://hecate.ia.forth.gr/catalog/hecate.csv",
        "https://hecate.ia.forth.gr/download/hecate.csv",
        "https://hecate.ia.forth.gr/downloads/hecate.csv"
    ]
    for u in urls_to_try:
        try:
            r = requests.get(u, timeout=15)
            tried.append((u, r.status_code))
            if r.status_code == 200 and len(r.content) > 200:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return True, u
        except Exception:
            continue
    return False, tried

def download_fallback_glade(dest_path):
    import requests
    print("Downloading fallback GLADE+ (this is large — if you want a smaller sample edit MAX_GALAXIES).")
    r = requests.get(GLADE_PLUS_URL, stream=True, timeout=60)
    if r.status_code == 200:
        # GLADE+ is an ASCII text file; we'll save locally
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
        return True
    return False

def ensure_catalog(path=CATALOG_LOCAL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"Using existing catalog at {path}")
        return path
    # try HECATE first
    try:
        ok, info = try_download_hecate(path)
    except Exception as e:
        ok = False
        info = str(e)
    if ok:
        print("Downloaded HECATE from", info)
        return path
    print("HECATE automatic download failed; trying GLADE+ fallback...")
    ok2 = download_fallback_glade(path)
    if ok2:
        print("Downloaded GLADE+ fallback to", path)
        return path
    raise RuntimeError("Failed to download both HECATE and GLADE+ catalogs. Please download manually and set CATALOG_LOCAL.")

# ----------------- Catalog parsing -----------------
def parse_catalog_into_centers(catalog_path, max_galaxies=MAX_GALAXIES):
    """
    Read CSV/txt and detect RA, Dec and distance columns. Return:
      centers (M,3) in AU, galaxy_masses (M,) in solar masses, names list
    The function auto-detects common column names.
    """
    # try reading with pandas; for GLADE+ space-delimited txt we try delim_whitespace if CSV fails
    try:
        df = pd.read_csv(catalog_path)
    except Exception:
        try:
            df = pd.read_csv(catalog_path, delim_whitespace=True, comment='#', header=None)
            # try to load columns if possible - fallback to raw columns
        except Exception as e:
            raise RuntimeError("Failed to read catalog file: " + str(e))

    # auto-detect RA/Dec columns
    ra_candidates = [c for c in df.columns if c.lower() in ('ra', 'ra_deg', 'ra_deg.')]
    dec_candidates = [c for c in df.columns if c.lower() in ('dec', 'dec_deg', 'dec_deg.')]
    dist_candidates = [c for c in df.columns if c.lower() in ('d','dist','distance','dist_mpc','dist_pc','d_mpc','d_pc')]
    mass_candidates = [c for c in df.columns if c.lower() in ('stellar_mass','mass','mstar','logmstar','log_mass','log_mass_stellar','m')]

    if not ra_candidates or not dec_candidates:
        # try columns named 'RA'/'DEC' in uppercase variants
        for c in df.columns:
            if c.strip().lower().startswith('ra'):
                ra_candidates.append(c)
            if c.strip().lower().startswith('dec'):
                dec_candidates.append(c)

    if not dist_candidates:
        # check for column named 'd' (HECATE uses 'd' in Mpc)
        for c in df.columns:
            if c.strip().lower() == 'd' or 'dist' in c.lower():
                dist_candidates.append(c)

    # fallback defaults
    ra_col = ra_candidates[0] if ra_candidates else None
    dec_col = dec_candidates[0] if dec_candidates else None
    dist_col = dist_candidates[0] if dist_candidates else None
    mass_col = mass_candidates[0] if mass_candidates else None

    centers = []
    masses = []
    names = []
    rows = df.shape[0]
    for idx in range(min(rows, max_galaxies*10)):  # scan up to a larger window to find good rows
        if ra_col is None or dec_col is None:
            break
        try:
            ra = float(df.iloc[idx][ra_col])
            dec = float(df.iloc[idx][dec_col])
            # distance
            if dist_col is not None:
                dist_val = df.iloc[idx][dist_col]
                if pd.isna(dist_val):
                    # skip
                    continue
                # guess units: if dist_col contains 'mpc' or 'Mpc' treat as Mpc
                if 'mpc' in str(dist_col).lower() or (isinstance(dist_val, (int,float)) and dist_val < 100):
                    # small values <100 probably Mpc (HECATE 'd' is in Mpc)
                    dist_pc = float(dist_val) * 1e6
                else:
                    dist_pc = float(dist_val)
            else:
                # fallback: assume 1 Mpc
                dist_pc = 1e6

            # convert to AU
            r_au = dist_pc * AU_PER_PC
            # cartesian
            ra_rad = math.radians(ra)
            dec_rad = math.radians(dec)
            x = r_au * math.cos(dec_rad) * math.cos(ra_rad)
            y = r_au * math.cos(dec_rad) * math.sin(ra_rad)
            z = r_au * math.sin(dec_rad)
            centers.append(np.array([x,y,z]))
            # mass detection
            if mass_col is not None:
                mval = df.iloc[idx][mass_col]
                if pd.isna(mval):
                    mass = 1e11
                else:
                    try:
                        # if log mass given (e.g., 'logmstar') detect via column name
                        if 'log' in str(mass_col).lower() or str(mass_col).lower().startswith('log'):
                            mass = 10**(float(mval))
                        else:
                            mass = float(mval)
                            # if mass seems tiny (<1e6) maybe it's log -> fallback
                            if mass < 1e6:
                                # assume it's log mass
                                mass = 10**mass
                    except Exception:
                        mass = 1e11
            else:
                mass = 1e11
            masses.append(mass)
            names.append(str(df.columns[0]) + f"_{idx}" if 'name' not in df.columns else str(df.iloc[idx].get('name', f'gal_{idx}')))
            if len(centers) >= max_galaxies:
                break
        except Exception:
            continue

    if len(centers) == 0:
        raise RuntimeError("No valid galaxy rows found in catalog. Inspect file and column names (RA/Dec/distance).")

    centers = np.vstack(centers)
    masses = np.array(masses)
    return centers, masses, names

# ----------------- physics & integrator (same approach as before) -----------------
if NUMBA:
    njit_decorator = njit
else:
    def njit_decorator(f):
        return f

@njit_decorator
def pairwise_accelerations(positions, masses, accelerations):
    N = positions.shape[0]
    for i in range(N):
        accelerations[i,0] = 0.0
        accelerations[i,1] = 0.0
        accelerations[i,2] = 0.0
    for i in range(N):
        xi, yi, zi = positions[i,0], positions[i,1], positions[i,2]
        mi = masses[i]
        axi = 0.0; ayi = 0.0; azi = 0.0

        for j in range(i+1, N):
            dx = positions[j,0] - xi
            dy = positions[j,1] - yi
            dz = positions[j,2] - zi
            dist_sqr = dx*dx + dy*dy + dz*dz
            if dist_sqr < 1e-20:
                continue
            dist_cubed = dist_sqr * math.sqrt(dist_sqr)
            factor = G / dist_cubed
            fx = dx * factor
            fy = dy * factor
            fz = dz * factor
            mj = masses[j]
            axi += fx * mj
            ayi += fy * mj
            azi += fz * mj
            accelerations[j,0] -= fx * mi
            accelerations[j,1] -= fy * mi
            accelerations[j,2] -= fz * mi
        accelerations[i,0] += axi
        accelerations[i,1] += ayi
        accelerations[i,2] += azi
    return accelerations

@njit_decorator
def rk4_step(positions, velocities, masses, dt, acc_buf):
    N = positions.shape[0]
    pairwise_accelerations(positions, masses, acc_buf)
    k1_v = acc_buf.copy()
    k1_x = velocities.copy()

    pos_tmp = positions + 0.5 * dt * k1_x
    vel_tmp = velocities + 0.5 * dt * k1_v
    pairwise_accelerations(pos_tmp, masses, acc_buf)
    k2_v = acc_buf.copy()
    k2_x = vel_tmp.copy()

    pos_tmp = positions + 0.5 * dt * k2_x
    vel_tmp = velocities + 0.5 * dt * k2_v
    pairwise_accelerations(pos_tmp, masses, acc_buf)
    k3_v = acc_buf.copy()
    k3_x = vel_tmp.copy()

    pos_tmp = positions + dt * k3_x
    vel_tmp = velocities + dt * k3_v
    pairwise_accelerations(pos_tmp, masses, acc_buf)
    k4_v = acc_buf.copy()
    k4_x = vel_tmp.copy()

    for i in range(N):
        for k in range(3):
            positions[i,k] += dt * (k1_x[i,k] + 2*k2_x[i,k] + 2*k3_x[i,k] + k4_x[i,k]) / 6.0
            velocities[i,k] += dt * (k1_v[i,k] + 2*k2_v[i,k] + 2*k3_v[i,k] + k4_v[i,k]) / 6.0

# ----------------- spawn & simulate -----------------
def spawn_bodies(centers, galaxy_masses, n_per=N_BODIES_PER_GAL, spawn_sigma=SPAWN_SIGMA_AU):
    M = centers.shape[0]
    Ntot = M * n_per
    pos = np.zeros((Ntot,3))
    vel = np.zeros((Ntot,3))
    masses = np.ones(Ntot)
    body_gal = np.zeros(Ntot, dtype=np.int32)
    idx = 0
    rng = np.random.RandomState(1234)
    for g in range(M):
        cen = centers[g]
        for i in range(n_per):
            r_offset = rng.normal(scale=spawn_sigma, size=3)
            pos[idx] = cen + r_offset
            v_rand = rng.normal(scale=1e-2, size=3)
            rmag = np.linalg.norm(r_offset) + 1e-8
            bias_mag = 0.05 * math.sqrt(galaxy_masses[g] / (1e10)) / (rmag/1e5 + 1e-6)
            bias = - (r_offset / rmag) * bias_mag
            vel[idx] = v_rand + bias
            masses[idx] = 1.0
            body_gal[idx] = g
            idx += 1
    return pos, vel, masses, body_gal

def simulate_from_catalog(catalog_path):
    centers, gal_masses, names = parse_catalog_into_centers(catalog_path, max_galaxies=MAX_GALAXIES)
    M = centers.shape[0]
    print(f"debug: shape {centers.shape} shape[0] {centers.shape[0]}")
    print(f"Loaded {M} galaxy centers from catalog.")
    pos, vel, masses, body_gal = spawn_bodies(centers, gal_masses, n_per=N_BODIES_PER_GAL)
    N = pos.shape[0]
    print(f"Spawned {N} bodies ({N_BODIES_PER_GAL} per galaxy).")

    gal_pos = centers.copy()
    gal_vel = np.zeros_like(gal_pos)
    gal_mass = gal_masses.copy()
    print(f"debug: shape {gal_pos.shape} shape[0] {gal_pos.shape[0]}")
    print(f"debug: shape {gal_vel.shape} shape[0] {gal_vel.shape[0]}")
    print(f"debug: shape {gal_mass.shape} shape[0] {gal_mass.shape[0]}")

    acc_buf = np.zeros_like(pos)
    gal_acc_buf = np.zeros_like(gal_pos)
    print(f"debug: shape {acc_buf.shape} shape[0] {acc_buf.shape[0]}")
    print(f"debug: shape {gal_acc_buf.shape} shape[0] {gal_acc_buf.shape[0]}")

    save_times = np.arange(0.0, TIME_YEARS+1e-9, SAVE_INTERVAL_YEARS)
    n_snap = save_times.shape[0]
    snapshots_pos = np.zeros((n_snap, N, 3))
    snapshots_vel = np.zeros((n_snap, N, 3))
    snapshots_pos[0] = pos
    snapshots_vel[0] = vel
    t = 0.0
    dt = INITIAL_DT

    for s_idx in range(1, n_snap):
        target_t = save_times[s_idx]
        while t < target_t - 1e-12:
            pairwise_accelerations(pos, masses, acc_buf)
            print("pass 1")
            print(f"debug: shape {gal_pos.shape} shape[0] {gal_pos.shape[0]}")
            print(f"debug: shape {gal_vel.shape} shape[0] {gal_vel.shape[0]}")
            print(f"debug: shape {gal_mass.shape} shape[0] {gal_mass.shape[0]}")
            # add galaxy center attraction
            for i in range(N):
                ax = ay = az = 0.0
                xi, yi, zi = pos[i]
                for g in range(M):
                    dx = gal_pos[g,0] - xi
                    dy = gal_pos[g,1] - yi
                    dz = gal_pos[g,2] - zi
                    dsq = dx*dx + dy*dy + dz*dz + 1e-12
                    inv_r3 = 1.0 / (dsq * math.sqrt(dsq))
                    factor = G * gal_mass[g] * inv_r3
                    ax += dx * factor
                    ay += dy * factor
                    az += dz * factor
                acc_buf[i,0] += ax
                acc_buf[i,1] += ay
                acc_buf[i,2] += az

            # adapt dt simple heuristic
            max_acc = np.max(np.sqrt((acc_buf**2).sum(axis=1))) + 1e-12
            min_dist = 1e30
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist:
                        min_dist = d
            suggested_dt = 0.2 * math.sqrt(min_dist / (max_acc + 1e-12))
            dt = max(1e-6, min(1e3, suggested_dt))

            rk4_step(pos, vel, masses, dt, acc_buf)
            pairwise_accelerations(gal_pos, gal_mass, gal_acc_buf)
            rk4_step(gal_pos, gal_vel, gal_mass, dt, gal_acc_buf)
            print("pass 2")
            print(f"debug: shape {gal_pos.shape} shape[0] {gal_pos.shape[0]}")
            print(f"debug: shape {gal_vel.shape} shape[0] {gal_vel.shape[0]}")
            print(f"debug: shape {gal_mass.shape} shape[0] {gal_mass.shape[0]}")
            t += dt

            # transfers
            tidal_radius = TIDAL_RADIUS_FACTOR * SPAWN_SIGMA_AU
            for i in range(N):
                gidx = body_gal[i]
                rmag = np.linalg.norm(pos[i] - gal_pos[gidx])
                if rmag > tidal_radius:
                    nearest_g = gidx
                    nearest_d = rmag
                    for g in range(M):
                        if g == gidx: continue
                        d = np.linalg.norm(pos[i] - gal_pos[g])
                        if d < nearest_d:
                            nearest_d = d
                            nearest_g = g
                    if nearest_g != gidx:
                        body_gal[i] = nearest_g
                        gal_mass[nearest_g] += masses[i]
                        gal_mass[gidx] -= masses[i]

            # merges
            merged = False
            for a in range(M):
                for b in range(a+1, M):
                    d = np.linalg.norm(gal_pos[a] - gal_pos[b])
                    if d < MERGE_DIST_AU:
                        total_mass = gal_mass[a] + gal_mass[b]
                        new_pos = (gal_mass[a]*gal_pos[a] + gal_mass[b]*gal_pos[b]) / total_mass
                        new_vel = (gal_mass[a]*gal_vel[a] + gal_mass[b]*gal_vel[b]) / total_mass
                        gal_pos[a] = new_pos
                        gal_vel[a] = new_vel
                        gal_mass[a] = total_mass
                        for i in range(N):
                            if body_gal[i] == b:
                                body_gal[i] = a
                        if b != M-1:
                            gal_pos[b] = gal_pos[M-1].copy()
                            gal_vel[b] = gal_vel[M-1].copy()
                            gal_mass[b] = gal_mass[M-1]
                            for i in range(N):
                                if body_gal[i] == M-1:
                                    body_gal[i] = b
                        M -= 1
                        gal_pos = gal_pos[:M]
                        gal_vel = gal_vel[:M]
                        gal_mass = gal_mass[:M]
                        merged = True
                        break
                if merged:
                    break

            # split logic
            for g in range(M):
                assigned = np.where(body_gal == g)[0]
                if assigned.size == 0:
                    continue
                distances = np.linalg.norm(pos[assigned] - gal_pos[g], axis=1)
                split_radius = 3.0 * SPAWN_SIGMA_AU
                frac_out = np.sum(distances > split_radius) / assigned.size
                if frac_out > SPLIT_FRAC and gal_mass[g] > 2e6:
                    esc_idxs = assigned[distances > split_radius]
                    centroid = pos[esc_idxs].mean(axis=0)
                    new_mass = np.sum(masses[esc_idxs])
                    gal_mass[g] -= new_mass
                    gal_pos = np.vstack([gal_pos, centroid])
                    gal_vel = np.vstack([gal_vel, np.zeros(3)])
                    gal_mass = np.concatenate([gal_mass, np.array([new_mass])])
                    new_g = gal_pos.shape[0]-1
                    for i in esc_idxs:
                        body_gal[i] = new_g
                    M += 1

        snapshots_pos[s_idx] = pos
        snapshots_vel[s_idx] = vel
        pct = int(100.0 * (s_idx) / (n_snap-1))
        print(f"{pct}% complete (t = {t:.2f} yr)")

    os.makedirs("data", exist_ok=True)
    np.savez("data/galaxy_sim_hecate.npz",
             body_positions = snapshots_pos,
             body_velocities = snapshots_vel,
             body_masses = masses,
             body_galaxy = body_gal,
             galaxy_positions = gal_pos,
             galaxy_masses = gal_mass,
             times = save_times)
    print("Saved data/galaxy_sim_hecate.npz")

# ----------------- main -----------------
if __name__ == "__main__":
    try:
        catalog_file = ensure_catalog()
    except Exception as e:
        print("Catalog download failed:", e)
        sys.exit(1)
    t0 = perf_counter()
    simulate_from_catalog(catalog_file)
    t1 = perf_counter()
    print("Elapsed (s):", t1-t0)
