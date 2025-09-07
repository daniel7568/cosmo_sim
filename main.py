#!/usr/bin/env python3
"""
lunar_eclipse_sim.py

Creates a 2D side-view animation (GIF) of Sun-Earth-Moon:
- Moon orbits Earth with ~5.145° inclination and nodal precession (~18.6 years).
- Earth's umbra at lunar distance is computed; frames where the Moon is inside umbra are detected.
- Saves GIF to ./lunar_eclipse_sim.gif

Dependencies:
    pip install numpy matplotlib pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
import math

# -------------------------
# Physical constants (km)
Re = 6371.0              # Earth radius
Rs = 696340.0            # Sun radius
d_SE = 149600000.0       # Sun-Earth distance
d_EM = 384400.0          # Earth-Moon distance

# Units: 1 Moon-distance = 1 unit
scale = d_EM
Re_u = Re / scale
Rs_u = Rs / scale
d_SE_u = d_SE / scale

# Moon orbit parameters
inc_deg = 5.145         # inclination w.r.t ecliptic (degrees)
inc = np.deg2rad(inc_deg)
T_moon = 27.321661      # sidereal period (days)
n_deg_day = 360.0 / T_moon

# Nodal precession (regression) ~18.6 years (period)
precession_years = 18.6
precession_deg_per_year = -360.0 / precession_years
precession_deg_per_day = precession_deg_per_year / 365.25
precession_rad_per_day = np.deg2rad(precession_deg_per_day)

# Umbra cone length and umbra radius at Moon distance (approx)
L_umbra = Re * d_SE / (Rs - Re)    # km
r_umbra_km = Re * (1 - (d_EM / L_umbra))
r_umbra_u = r_umbra_km / scale

# Simulation time and frames
years_to_sim = 5.0    # compress a few years so precession is visible
days_total = years_to_sim * 365.25
n_frames = 400
t_array = np.linspace(-10.0, days_total + 10.0, n_frames)  # days

# Sun position on negative x-axis (in Earth-centered units)
sun_pos = np.array([-d_SE_u, 0.0, 0.0])

# Initial node angle (Omega)
Omega0 = 0.0

def moon_position(t_days):
    """
    Return moon position in Earth-centered coordinates (units of lunar distance).
    Uses: rotate by Omega around z, then incline by inc about x (Rz * Rx).
    """
    theta = np.deg2rad(n_deg_day * t_days)        # mean anomaly angle (orbital phase)
    Omega = Omega0 + precession_rad_per_day * t_days
    r = 1.0  # approx circular orbit with radius = 1 unit (Moon distance)
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    z_orb = 0.0

    # Rx(inc)
    cosi = np.cos(inc); sini = np.sin(inc)
    x1 = x_orb
    y1 = y_orb * cosi - z_orb * sini
    z1 = y_orb * sini + z_orb * cosi

    # Rz(Omega)
    cosO = np.cos(Omega); sinO = np.sin(Omega)
    x = cosO * x1 - sinO * y1
    y = sinO * x1 + cosO * y1
    z = z1
    return np.array([x, y, z])

# Precompute moon positions and umbra membership
moon_pos = np.array([moon_position(t) for t in t_array])

sun_dir = (sun_pos - np.array([0.0, 0.0, 0.0]))
sun_dir = sun_dir / np.linalg.norm(sun_dir)
shadow_axis = -sun_dir  # direction from Earth toward the shadow (points roughly +x)

in_umbra = []
for r in moon_pos:
    proj = np.dot(r, shadow_axis)
    perp = r - proj * shadow_axis
    dist_axis = np.linalg.norm(perp)
    # Moon must be behind Earth relative to Sun (proj > 0) and within umbra radius
    cond = (proj > 0.0) and (dist_axis < r_umbra_u)
    in_umbra.append(cond)
in_umbra = np.array(in_umbra)

# -------------------------
# Create animation (side view: x vs z)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal', 'box')
ax.set_title('Moon, Earth and Umbra (side view). Time compressed: {:.1f} years'.format(years_to_sim))

# Sun marker (left)
ax.plot([sun_pos[0]], [0.0], marker='o', markersize=10)
ax.text(sun_pos[0], -0.1, 'Sun', fontsize=8, ha='center')

# Earth (drawn larger for visibility)
earth_circle = Circle((0.0, 0.0), Re_u * 4.0, fill=False, linewidth=1.2)
ax.add_patch(earth_circle)
ax.text(0.02, -0.05, 'Earth (not to scale)', fontsize=8)

# Umbra circle at Moon distance (projected onto side view)
shadow_center = shadow_axis * 1.0
umbra_circle = Circle((shadow_center[0], shadow_center[2]), r_umbra_u, fill=False, linestyle='--')
ax.add_patch(umbra_circle)
ax.text(shadow_center[0] + 0.02, shadow_center[2], 'Umbra @ Moon dist', fontsize=8)

# Moon plot handle
moon_marker, = ax.plot([], [], marker='o', markersize=8)

# Info text
time_text = ax.text(-1.4, 1.0, '', fontsize=9)
status_text = ax.text(-1.4, 0.9, '', fontsize=9)

def init():
    moon_marker.set_data([], [])
    time_text.set_text('')
    status_text.set_text('')
    return moon_marker, time_text, status_text

def update(frame):
    r = moon_pos[frame]
    x = r[0]; z = r[2]
    moon_marker.set_data([x], [z])   # <-- FIX HERE
    Omega = Omega0 + precession_rad_per_day * t_array[frame]
    node_deg = np.rad2deg(Omega) % 360.0
    t_days = t_array[frame]
    time_text.set_text('t = {:.1f} days\nNode (Ω) = {:.1f}°'.format(t_days, node_deg))
    status = 'In umbra' if in_umbra[frame] else 'Not in umbra'
    status_text.set_text(status)
    return moon_marker, time_text, status_text


anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=n_frames, interval=30, blit=True)

# # Save as GIF
# gif_path = 'lunar_eclipse_sim.gif'
# from matplotlib.animation import PillowWriter
# writer = PillowWriter(fps=20)
# anim.save(gif_path, writer=writer)


anim.save('lunar_eclipse_sim.mp4', fps=30, writer='ffmpeg')

