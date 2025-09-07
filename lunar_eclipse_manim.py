from manim import *
import numpy as np

class LunarEclipseScene(Scene):
    def construct(self):
        # Background stars
        stars = VGroup()
        np.random.seed(1)
        for _ in range(200):
            x, y = np.random.uniform(-7, 7), np.random.uniform(-4, 4)
            star = Dot(point=[x, y, 0], radius=0.01, color=WHITE)
            stars.add(star)
        self.add(stars)

        # Moon (as a circle)
        moon_radius = 1
        moon = Circle(radius=moon_radius, color=WHITE, fill_opacity=1).shift(DOWN*0.5)
        self.add(moon)

        # Shadow (umbra = black, penumbra = gray)
        shadow_radius = 2.5   # radius of shadow circle at moon's distance
        umbra = Circle(radius=shadow_radius*0.6, color=BLACK,
                       fill_opacity=0.9).shift(LEFT*6)
        penumbra = Circle(radius=shadow_radius, color=GRAY,
                          fill_opacity=0.5).shift(LEFT*6)

        self.add(penumbra, umbra)

        # Animate shadow crossing the Moon
        total_shift = 12  # how far shadow moves
        duration = 20     # seconds of animation

        self.play(
            penumbra.animate.shift(RIGHT*total_shift),
            umbra.animate.shift(RIGHT*total_shift),
            run_time=duration,
            rate_func=linear
        )

        # Hold final frame for 2 seconds
        self.wait(2)
