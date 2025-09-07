import numpy as np
import plotly.graph_objects as go

# --- Load simulation data ---
data = np.load("data/galaxy_sim_hecate.npz")
positions = data["body_positions"]  # shape (T, N, 3)
times = data["times"]
T, N, _ = positions.shape

# --- Prepare Plotly frames ---
frames = []
for t in range(T):
    frames.append(go.Frame(
        data=[go.Scatter3d(
            x=positions[t, :, 0],
            y=positions[t, :, 1],
            z=positions[t, :, 2],
            mode='markers',
            marker=dict(size=3, color='blue')
        )],
        name=str(t)
    ))

# --- Initial figure ---
fig = go.Figure(
    data=[go.Scatter3d(
        x=positions[0, :, 0],
        y=positions[0, :, 1],
        z=positions[0, :, 2],
        mode='markers',
        marker=dict(size=3, color='blue')
    )],
    layout=go.Layout(
        title="Galaxy Simulation",
        scene=dict(
            xaxis_title="X [AU]",
            yaxis_title="Y [AU]",
            zaxis_title="Z [AU]"
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=50, redraw=True),
                                           fromcurrent=True, mode="immediate")])]
        )]
    ),
    frames=frames
)

fig.show()
