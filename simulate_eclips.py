#!/usr/bin/env python3
"""
simulate_eclipse.py

Physics-focused per-pixel lunar eclipse renderer.

Outputs:
 - frames/frame_###.png   (grayscale (0..255) PNG frames showing Moon disk as seen from Earth)
 - eclipse_frames.npz    (metadata: times_hours, frame_count, moon_plane_extent, etc.)

Usage:
    python simulate_eclipse.py

Dependencies:
    pip install numpy pillow tqdm
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

# ---------------------------
# Physical constants (km)
Re = 6371.0          # Earth radius
Rm = 1737.4          # Moon radius
Rs = 696340.0        # Sun radius
d_SE = 149600000.0   # Sun-Earth distance (approx)
d_EM = 384400.0      # Earth-Moon distance (mean)

# ---------------------------
# Simulation time window and resolution
hours_window = 6.0        # total simulated hours (typical eclipse ~3-6 hours); center is eclipse mid
frames_per_hour = 20      # temporal sampling; 20 -> 120 frames for 6 hours (~3 frames/min)
n_frames = int(hours_window * frames_per_hour)
times_hours = np.linspace(-hours_window/2, hours_window/2, n_frames)

# ---------------------------
# Moon-plane image geometry (units: km)
# We'll render a square image centered on the Moon with half-width = moon_radius * pad_factor
pad_factor = 1.6   # how much larger than the moon radius the image extents are
half_width_km = Rm * pad_factor
img_pixels = 512   # set to 512 for good quality; increase to 768/1024 if you want slower but sharper frames

# pixel coordinates in km, centered at moon center; positive X to right, positive Y up
xs = np.linspace(-half_width_km, half_width_km, img_pixels)
ys = np.linspace(-half_width_km, half_width_km, img_pixels)
X_km, Y_km = np.meshgrid(xs, ys)

# Precompute moon disk mask in these km coordinates
dist2 = X_km**2 + Y_km**2
moon_mask = dist2 <= (Rm**2)

# ---------------------------
# Geometry helpers
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

# Solve for real eclipse event center using approximate Keplerian orbit,
# or, for simplicity and robustness here, we will place the Moon so that
# at t=0 it is at a "near-full near-node" position and the shadow sweeps horizontally.
#
# For an accurate orbital simulator you can re-use the Kepler solver from earlier code.
# Here we choose an event-centered geometry: Moon center at (d_EM,0,0) in Earth-centered coords,
# and the shadow axis passes across with lateral offset sweep to produce an eclipse.
#
# But we want realistic distances for angular calculations, so we create a simple parametric sweep
# where the moon center vector in Earth-centered coords moves slightly in Y so that the shadow center
# relative to moon crosses the disk. This simplifies the demonstration but keeps angular correctness.

# Place Earth at origin, Sun along -X at (-d_SE, 0, 0)
sun_pos = np.array([-d_SE, 0.0, 0.0])

# We'll parametrize moon center vector: base at +X d_EM, plus small Y motion to simulate pass through shadow
# maximum lateral shift (km) chosen to ensure full-to-no eclipse across the run
max_lateral_km = Rm * 3.2  # adjust to increase / decrease maximum offset (3.2*Rm typically crosses fully)
moon_centers = np.zeros((n_frames, 3))
for i, th in enumerate(times_hours):
    frac = (i / max(1, n_frames-1)) * 2 - 1.0   # -1..1 across frames
    # We'll move moon center in Y direction
    moon_centers[i] = np.array([d_EM, frac * max_lateral_km, 0.0])

# If you want to replace the above simple sweep with real orbital positions, swap moon_centers with km positions.

# ---------------------------
# Function: circle overlap area for two circles (r1,r2) with center distance d (in same units)
def circle_overlap_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2
    # general case
    a1 = r1*r1 * math.acos((d*d + r1*r1 - r2*r2) / (2*d*r1))
    a2 = r2*r2 * math.acos((d*d + r2*r2 - r1*r1) / (2*d*r2))
    a3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2)*(d + r1 - r2)*(d - r1 + r2)*(d + r1 + r2)))
    return a1 + a2 - a3

# ---------------------------
# For each pixel, we will:
# - compute its 3D position in ECI-like coords: p = moon_center + x*e1 + y*e2 (where e1,e2 basis chosen)
#   We choose e1 = +Y_world, e2 = +Z_world (simple because we keep moon plane orthogonal to X)
#   That maps pixel (X_km, Y_km) into vectors p_km = [d_EM, Y_km, X_km] (we place Z = X_km to have visible vertical)
#
# - compute vector to Earth center (v_e = -p_km), distance r_e = |v_e|, Earth angular radius = alpha_e = asin(Re / r_e)
# - compute vector to Sun center (v_s = sun_pos - p_km), distance r_s = |v_s|, Sun angular radius = alpha_s = asin(Rs / r_s)
# - compute angular separation gamma between v_e and v_s: gamma = arccos( dot(unit(v_e), unit(v_s)) )
# - treat the two disks on a tangent plane with radii alpha_e and alpha_s and centers separated by d_ang = gamma
#   compute circle overlap in angular units using circle_overlap_area; fraction_occluded = overlap_area / (pi * alpha_s^2)
# - brightness = 1 - fraction_occluded (clamped [0,1])
#
# This small-angle planar overlap approximation is very accurate for these small angular radii (~0.25°).

# Pre-flatten arrays for speed
Xflat = X_km.ravel()
Yflat = Y_km.ravel()
mask_flat = moon_mask.ravel()
npix = img_pixels * img_pixels

# Prepare output directory
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

# iterate frames and render
for idx in tqdm(range(n_frames), desc="Rendering frames"):
    mcenter = moon_centers[idx]            # km
    # prepare brightness array
    img = np.zeros((img_pixels, img_pixels), dtype=np.float32)

    # compute pixel world positions: p_km = mcenter + [0, x (km), y (km)] using axes mapping:
    # We'll map image X->physical Y and image Y->physical Z to create some vertical structure
    # So vector p = [mcenter_x, mcenter_y + Xflat, mcenter_z + Yflat]
    px = np.full(Xflat.shape, mcenter[0])   # X coordinate = radial distance
    py = mcenter[1] + Xflat                 # lateral (east-west)
    pz = mcenter[2] + Yflat                 # vertical (north-south)

    # vector to Earth center
    vx_e = -px; vy_e = -py; vz_e = -pz
    re = np.sqrt(vx_e*vx_e + vy_e*vy_e + vz_e*vz_e)

    # vector to Sun center
    sx = sun_pos[0] - px; sy = sun_pos[1] - py; sz = sun_pos[2] - pz
    rs = np.sqrt(sx*sx + sy*sy + sz*sz)

    # small safety: clamp ratios
    # angular radii
    # protect against asin domain errors
    with np.errstate(invalid='ignore'):
        alpha_e = np.arcsin(np.clip(Re / re, -0.9999999, 0.9999999))   # radians
        alpha_s = np.arcsin(np.clip(Rs / rs, -0.9999999, 0.9999999))   # radians

    # separation angle gamma between vectors v_e and v_s
    # dot = v_e dot v_s / (re * rs)
    dot = (vx_e * sx + vy_e * sy + vz_e * sz) / (re * rs)
    dot = np.clip(dot, -1.0, 1.0)
    gamma = np.arccos(dot)   # radians

    # now compute overlap area per pixel in angular units using vectorized loops in chunks
    alpha_s_sq = alpha_s * alpha_s
    # avoid division by zero
    # We'll compute overlap by iterating only over moon_mask pixels for speed
    indices = np.nonzero(mask_flat)[0]
    brightness = np.zeros(indices.shape[0], dtype=np.float32)

    for k_i, pix_idx in enumerate(indices):
        r_e_ang = alpha_e[pix_idx]
        r_s_ang = alpha_s[pix_idx]
        d_ang = gamma[pix_idx]

        if r_s_ang <= 0:
            frac = 0.0
        else:
            # overlap area
            area_overlap = circle_overlap_area(r_s_ang, r_e_ang, d_ang)
            # normalize by Sun area
            sun_area = math.pi * (r_s_ang**2)
            frac = float(area_overlap / sun_area) if sun_area > 0 else 0.0
            frac = min(max(frac, 0.0), 1.0)

        brightness[k_i] = 1.0 - frac   # 1 = full bright, 0 = fully eclipsed

    # assemble full image
    flat_img = np.zeros(mask_flat.shape, dtype=np.float32)
    flat_img[indices] = brightness
    flat_img = flat_img.reshape((img_pixels, img_pixels))

    # set outside-moon to black (sky) -> 0.0
    # Gamma-correct slightly for nicer appearance (optional)
    gamma_corr = 1.0
    img_display = np.clip(flat_img**(1.0/gamma_corr), 0.0, 1.0)

    # convert to uint8 grayscale [0..255], invert so 255 = bright
    arr8 = (img_display * 255.0).astype(np.uint8)

    # Save PNG; flip vertically since image coords row0 is top in many viewers; keep consistent with Manim origin
    img_pil = Image.fromarray(arr8)
    fname = os.path.join(out_dir, f"frame_{idx:04d}.png")
    img_pil.save(fname)

# Save metadata
np.savez("eclipse_frames.npz",
         times_hours = times_hours,
         n_frames = n_frames,
         img_pixels = img_pixels,
         half_width_km = half_width_km,
         frames_dir = out_dir,
         Rm = Rm,
         Re = Re,
         Rs = Rs,
         d_EM = d_EM,
         d_SE = d_SE)

print("Done. Wrote", n_frames, "frames to", out_dir, "and eclipse_frames.npz")
