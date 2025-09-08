# lunar_eclipse_manim.py
from manim import *
import numpy as np
import os

class LunarEclipseScene(Scene):
    def construct(self):
        data = np.load("eclipse_frames.npz", allow_pickle=True)
        frames_dir = str(data["frames_dir"])
        n_frames = int(data["n_frames"])
        times = data["times_hours"]
        img_pixels = int(data["img_pixels"])
        half_width_km = float(data["half_width_km"])
        Rm = float(data["Rm"])

        # Visual scale: set moon radius = 1.0 units in manim coordinates
        moon_visual_radius = 1.0
        km_to_unit = moon_visual_radius / Rm

        # Load a few frames to get aspect ratio and build ImageMobjects lazily
        image_paths = [os.path.join(frames_dir, f"frame_{i:04d}.png") for i in range(n_frames)]
        # Create ImageMobject for the first frame and scale to visual moon radius
        first_img = ImageMobject(image_paths[0])
        # Scale so that image width across moon corresponds to 2*moon_visual_radius (full moon diameter)
        # The pixel diameter in the image that correspond to moon diameter is roughly: 2*Rm / (2*half_width_km) * img_pixels
        # But easier: scale so that image height mapped to 2*moon_visual_radius when radius in pixels equals Rm pixels mapping.
        # We compute mapping: Rm pixels in image corresponds to moon_visual_radius units; find scale factor
        # Compute pixel radius in image: since image covers 2*half_width_km and has img_pixels pixels,
        # pixel_km = (2*half_width_km) / img_pixels  => Rm pixels corresponds to Rm / pixel_km pixels = Rm * img_pixels / (2*half_width_km)
        pixel_radius = Rm * img_pixels / (2.0 * half_width_km)
        # width_in_pixels across whole image
        img_width_px = first_img.get_width()
        # scale factor to map pixel_radius -> moon_visual_radius
        # In manim, Img.get_width() returns width in scene units; after creation it has default width 2
        # So compute scale to get desired moon size.
        # Simpler approach: set image width so that pixel_radius pixels correspond to moon_visual_radius units:
        # desired_width_units = (img_width_px / pixel_radius) * moon_visual_radius
        # But manim scales ImageMobject by .scale(factor) where initial width is first_img.width
        # We'll compute target_width = (img_width_px / pixel_radius) * moon_visual_radius
        # then scale factor = target_width / first_img.width
        # To avoid depending on pixel metrics, we use an empirical approach: scale the image so that its width equals 2*moon_visual_radius * (img_pixels/(2*Rm_pixels))
        # Simpler and robust: compute the desired width in manim units as:
        desired_width = 2.0 * moon_visual_radius * (img_pixels / (2.0 * pixel_radius))  # simplifies to = 2*moon_visual_radius * (img_pixels / (2*pixel_radius))
        # But algebra cancels; instead, we'll directly scale using ratio of pixel_radius to image pixel width:
        # image pixel width = img_pixels (square). So pixel_radius proportion of image width = pixel_radius / img_pixels
        proportion = pixel_radius / img_pixels
        target_image_width = (2.0 * moon_visual_radius) / (2.0 * proportion)  # simplifies to moon_visual_radius / proportion
        scale_factor = target_image_width / first_img.width

        first_img.scale(scale_factor)
        # place at center
        first_img.move_to(ORIGIN)
        self.add(first_img)
        # Add title and time label
        title = Text("Lunar Eclipse — Earth-sky view", font_size=36).to_edge(UP)
        time_label = Text("", font_size=24).to_edge(UR)
        self.add(title, time_label)

        # Create ImageMobjects for all frames but keep them off-screen (or create on the fly to reduce memory)
        # We'll display frames by replacing the previous image with the next one for smooth playback
        current = first_img
        fps = 25  # playback fps; choose to match frames_per_hour from simulation mapping
        # compute total runtime mapping: n_frames -> desired seconds; pick e.g., 20 seconds final video
        total_seconds = 20.0
        per_frame_time = total_seconds / max(1, n_frames - 1)

        for i, path in enumerate(image_paths[1:], start=1):
            img_obj = ImageMobject(path)
            img_obj.scale(scale_factor)
            img_obj.move_to(ORIGIN)
            # Transition: cross-fade for smoothness
            self.play(ReplacementTransform(current, img_obj), run_time=per_frame_time, rate_func=linear)
            current = img_obj
            # update time label
            time_label.become(Text(f"t = {times[i]:.2f} h", font_size=22).to_edge(UR))

        # hold final frame
        self.wait(2)
