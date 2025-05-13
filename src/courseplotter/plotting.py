from dataclasses import dataclass
from typing import Optional

import contextily as cx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.courseplotter.data import calculate_plotting_coordinates


def get_randomness_size(distances: pd.Series, bins: int = 100, scale: int = 2500) -> np.ndarray:
    density_values, bin_edges = np.histogram(distances, bins=bins, density=False)
    density_values[density_values > scale] = scale

    bin_indices = np.digitize(distances, bin_edges) - 1
    bin_indices[distances < bin_edges[0]] = -1
    bin_indices[distances >= bin_edges[-1]] = bins - 1

    return density_values[bin_indices] / scale


@dataclass
class Theme:
    plot_scale: int = 1
    plot_width: int = 16
    plot_height: int = 9
    plot_zoom: float = 10
    background_colour: str = "black"
    font_colour: str = "white"
    randomness_scale: float = 1


class CoursePlotter:
    def __init__(self, timings: pd.DataFrame, course: pd.DataFrame, theme: Theme) -> None:
        self.timings = timings
        self.course = course
        self.positions = None

        # Plot componenets
        self.theme = theme
        self.fig = None
        self.ax = None
        self.timer_text = None
        self.course_plot = None
        self.colour_scatter = None
        self.white_scatter = None
        self.plot_size = None

    def calculate_positions(self, frequency_s: int) -> None:
        self.positions = calculate_plotting_coordinates(timings=self.timings, course=self.course, frequency=frequency_s)

    def create_figure(self) -> None:
        if self.positions is None:
            raise ValueError("Positions must be calculated first: Run `CoursePlotter.calculate_positions`")

        self.fig, self.ax = plt.subplots(figsize=(self.theme.plot_width, self.theme.plot_height))
        self.ax.grid(False)
        self.ax.axis("off")
        self.ax.set_aspect("equal", "box")

        # Apply configuration:
        self._set_figure_colours()
        self._set_axis_limits()
        self._add_basemap()
        self._add_timer_box()

    def create_plots(self) -> None:
        self.course_plot = self.ax.plot(
            self.course["longitude"], self.course["latitude"], alpha=0.5, c="grey", zorder=1
        )
        self.colour_scatter = self.ax.scatter(
            [], [], s=5, alpha=0.05, c=[], cmap="RdYlBu", vmin=0, vmax=42200, zorder=2
        )
        self.white_scatter = self.ax.scatter([], [], s=5, alpha=0.025, c="white", zorder=3)

    def plot_data(self, time_s: Optional[int] = None) -> plt.Figure:
        all = self.positions
        df = all.loc[all["time"] == time_s].copy()

        base_scale = self.plot_size * 0.0025
        df["weights"] = get_randomness_size(df["distance"], 100, 150) * base_scale * self.theme.randomness_scale
        random_x = np.random.normal(scale=df["weights"], size=len(df))
        random_y = np.random.normal(scale=df["weights"], size=len(df))

        # Coloured dots:
        df["longitude_noise"] = df["longitude"] + random_x
        df["latitude_noise"] = df["latitude"] + random_y
        coordinates = df[["longitude_noise", "latitude_noise"]].values
        self.colour_scatter.set_offsets(coordinates)
        self.colour_scatter.set_array(df["distance"])

        # White dots:
        df["longitude_noise"] = df["longitude"] + (random_x * 0.75)
        df["latitude_noise"] = df["latitude"] + (random_y * 0.75)
        coordinates = df[["longitude_noise", "latitude_noise"]].values
        self.white_scatter.set_offsets(coordinates)

        # Update timer:
        hours = int(time_s // 3600)
        minutes = int((time_s % 3600) // 60)
        self.timer_text.set_text(f"{hours:02d}:{minutes:02d}")

        return self.fig

    def animate_plot(self, filename: str) -> None:
        frames = self.positions["time"].unique()
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self.update_animation,
            frames=frames,
            interval=400,  # milliseconds between frames (e.g., 200ms)
            blit=True,  # Set to False if updating size/color per frame causes issues
            repeat=True,
        )

        with tqdm(total=len(frames), desc="Saving video") as pbar:

            def update_func(_i, _n) -> None:
                pbar.update()

            ani.save(filename, fps=24, dpi=72, progress_callback=update_func)

    def update_animation(self, frame_seconds: int) -> None:
        """
        Updates the scatter plot for each frame (each unique second).

        Args:
            frame_seconds: The current 'time' value for this frame.

        Returns:
            A tuple containing the updated scatter plot PathCollection.
        """
        _ = self.plot_data(frame_seconds)

        return (self.colour_scatter, self.white_scatter)

    def _add_basemap(self) -> None:
        try:
            cx.add_basemap(self.ax, crs="EPSG:4326", source=cx.providers.CartoDB.DarkMatter)
        except Exception as e:
            raise Exception(
                f"Could not add basemap: {e}\nEnsure you have internet access and a basemap source is available."
            )

    def _set_figure_colours(self) -> None:
        self.fig.patch.set_facecolor(self.theme.background_colour)
        self.ax.set_facecolor(self.theme.background_colour)

    def _set_axis_limits(self) -> None:
        # Calculate the range of the course data in both dimensions
        lon_max = self.course["longitude"].max()
        lon_min = self.course["longitude"].min()
        lat_max = self.course["latitude"].max()
        lat_min = self.course["latitude"].min()
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        self.plot_size = max(lon_range, lat_range)

        # Calculate the center of the data's bounding box
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2

        # Get the plot aspect ratio (width / height)
        ratio = self.theme.plot_width / self.theme.plot_height
        if ratio <= 0:
            ratio = 1.0  # Default to square aspect ratio if invalid

        # Define margin in data units
        margin = 0.05
        half_lon_span = max(lon_range / 2, ratio * (lat_range / 2)) * (margin + 1)
        half_lat_span = half_lon_span / ratio * (margin + 1)

        # Calculate the new limits based on the center and the calculated half-spans
        new_lon_min = center_lon - half_lon_span
        new_lon_max = center_lon + half_lon_span
        new_lat_min = center_lat - half_lat_span
        new_lat_max = center_lat + half_lat_span

        # Set the plot limits
        self.ax.set_xlim(new_lon_min, new_lon_max)
        self.ax.set_ylim(new_lat_min, new_lat_max)

    def _add_timer_box(self) -> None:
        self.timer_text = self.ax.text(
            0.99,
            0.01,
            "",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=self.ax.transAxes,  # This makes the coordinates relative to the axes
            color=self.theme.font_colour,  # Set color for visibility
            fontsize=24,
            bbox=dict(boxstyle="round,pad=0.5", fc=self.theme.background_colour, alpha=0.5),
        )
