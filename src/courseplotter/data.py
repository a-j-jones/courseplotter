import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column name `{column}` is required")


@nb.njit
def interpolate_times(times: np.ndarray, course: np.ndarray, frequency_s: int) -> np.ndarray:
    """
    Interpolate positions on the course for a single competitor.

    Args:
        times (np.ndarray): Columns [distance, time] for a single competitor
        course (np.ndarray): Columns [distance, longitude, latitude]
        frequency_s (int): Frequency for interpolation in seconds

    Returns:
        np.ndarray: Interpolated positions of a competitor at the provided interval [seconds, distance, longitude, latitude]
    """
    out_size = (0, 4)

    if times.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    if times[0, 0] != 0.0:
        zero_start_record = np.zeros((1, times.shape[1]), dtype=times.dtype)
        times = np.vstack((zero_start_record, times))

    if course.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    r_dist = times[:, 0]
    r_secs = times[:, 1]
    c_dist = course[:, 0]

    if r_dist.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    c_secs = np.interp(c_dist, r_dist, r_secs)

    if times.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    max_time = times[-1, 1]
    if frequency_s <= 0:
        return np.empty(out_size, dtype=np.float64)

    p_secs = np.arange(0.0, max_time + frequency_s, float(frequency_s))

    if p_secs.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    c_lon = course[:, 1]
    c_lat = course[:, 2]

    if c_secs.shape[0] == 0:
        return np.empty(out_size, dtype=np.float64)

    p_lon = np.interp(p_secs, c_secs, c_lon)
    p_lat = np.interp(p_secs, c_secs, c_lat)
    p_dist = np.interp(p_secs, c_secs, c_dist)

    out_data = np.empty((p_secs.shape[0], 4), dtype=np.float64)
    out_data[:, 0] = p_secs
    out_data[:, 1] = p_dist
    out_data[:, 2] = p_lon
    out_data[:, 3] = p_lat

    return out_data


def apply_interpolation(
    group: pd.DataFrame,
    course_arr: np.ndarray,
    frequency_s: int,
    pbar: tqdm,
) -> np.ndarray:
    """
    Wrapper function to apply the interpolation logic to a group (DataFrame for one runner).
    """
    arr = group[["distance", "time"]].values
    out_cols = ["time", "distance", "longitude", "latitude"]

    if arr.shape[0] == 0:
        pbar.update()
        return pd.DataFrame(np.empty((0, 3)), columns=out_cols)

    df = pd.DataFrame(interpolate_times(arr, course_arr, frequency_s), columns=out_cols)
    pbar.update()

    return df


def calculate_plotting_coordinates(timings: pd.DataFrame, course: pd.DataFrame, frequency: int) -> pd.DataFrame:
    check_columns(timings, ["id", "distance", "time"])
    check_columns(course, ["distance", "longitude", "latitude"])

    course_array = course[["distance", "longitude", "latitude"]].values

    with tqdm(total=len(timings["id"].unique()), position=0) as pbar:
        interpolated_timings: pd.DataFrame = timings.groupby("id").apply(
            apply_interpolation, course_arr=course_array, frequency_s=frequency, pbar=pbar, include_groups=False
        )

    interpolated_timings = interpolated_timings.dropna()

    return interpolated_timings
