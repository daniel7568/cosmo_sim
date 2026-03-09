import taichi as ti #main tool
import taichi.math as tm #math functions and vector types
import numpy as np #for generating data

# Initialize Taichi with GPU support for better performances
ti.init(arch=ti.gpu)

#physical parameters
N = 100_000 #number of particles
m = 40000 #mass of each particle(unit less)
eps = 1e+0 #softening parameter
energy = 0.00000005 #energy parameter
dt = 0.001 #time step

#display parameters
center_x = 0
center_y = 0
zoom_speed = 1.1
min_zoom = 1.0
max_zoom = 5000.0
world_x = 400
world_y = 400

#Creating fields for the positions, velocities, and accelerations of the particles
positions = ti.Vector.field(2, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N)
accelerations = ti.Vector.field(2, dtype=ti.f32, shape=N)

#Creating a field for the positions of the particles to be displayed
display_pos = ti.Vector.field(2, dtype=ti.f32, shape=N)

#Generate random positions and velocities for the particles. The velocities are set to be perpendicular to the position vectors.
pos_np = np.random.rand(N,2)*500-250
mag = np.linalg.norm(pos_np, axis=1, keepdims=True)
vel_np = np.empty_like(pos_np)
vel_np[:,0] = -pos_np[:,1] * mag[::-1].T
vel_np[:,1] =  pos_np[:,0] * mag[::-1].T

#initialize the fields with the generated data. The velocities are scaled by the energy parameter to control the initial kinetic energy of the system.
velocities.from_numpy(vel_np*energy)
positions.from_numpy(pos_np)


@ti.func
def compute_accelerations():
    """
    Computes the forces acting on each particle due to the gravitational attraction of all other particles
    """
    #zero accelerations
    for i in range(N):
        accelerations[i] = 0

    ti.block_local(positions)
    #loop over all pairs of particles and compute the force between them. (applied only to one of them for parallelization)
    for i in range(N):
         for j in range(N):
            d = positions[j] - positions[i]
            dist2 = d.dot(d) + eps
            inv_r = ti.rsqrt(dist2)
            inv3 = inv_r * inv_r * inv_r
            accelerations[i] += m * d * inv3 #G = 1

@ti.kernel
def step():
    """
    Uses the leapfrog integration to simulate the motion of the particles
    """
    #half-kick
    compute_accelerations()
    for i in range(N):
        velocities[i] += 0.5 * dt * accelerations[i]

    #drift
    for i in range(N):
        positions[i] += dt * velocities[i]

    #half-kick
    compute_accelerations()
    for i in range(N):
        velocities[i] += 0.5 * dt * accelerations[i]

@ti.kernel
def to_display(center_x: float, center_y: float, world_x: float, world_y: float):
    """
    takes the position of the center and the zoom(world_x and world_y) and scales the positions to the display)
    """
    world = tm.vec2(world_x, world_y)
    center = tm.vec2(center_x, center_y)
    for i in positions:
        display_pos[i] = (positions[i] - center + world) / (2 * world)


#initialize the window and scene
window = ti.ui.Window("Simulation", res=(800,800))
canvas = window.get_canvas()
scene  = window.get_scene()

#main simulation loop that handles the simulation and user interaction
while window.running:

    #handle user input: hold mouse 1 to move camera, e to zoom in, q to zoom out
    cx, cy = window.get_cursor_pos()
    if window.is_pressed(ti.ui.LMB):
        dx = cx - last_cx
        dy = cy - last_cy
        if dx != 0 or dy != 0:
            change = True
            center_x -= dx * world_x
            center_y -= dy * world_y
    last_cx, last_cy = cx, cy
    if window.is_pressed('e'):  # mouse 4
        world_x /= zoom_speed
        world_y /= zoom_speed
        print(f"zoom: {world_x} {world_y}")
    elif window.is_pressed('q'):  # mouse 5
        world_x *= zoom_speed
        world_y *= zoom_speed
        print(f"zoom: {world_x} {world_y}")

    step() #physics code

    to_display(center_x, center_y, world_x, world_y) #scale data for display
    canvas.circles(display_pos, radius=0.001, color=(1.0, 1.0, 1.0)) #draw the particles
    window.show()



