import numba as nb
import numpy as np


@nb.njit
def interpolate_times(times: np.ndarray, course: np.ndarray, frequency_s: int) -> np.ndarray:
    """
    Interpolate positions on the course for a single competitor.
    - times: np.ndarray with columns [distance, time_s] for a single competitor.
    - course: np.ndarray with columns [course_dist, course_lon, course_lat].
    - frequency_s: The desired time frequency in seconds for the output.
    """
    if times.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    if times[0, 0] != 0.0:
        zero_start_record = np.zeros((1, times.shape[1]), dtype=times.dtype)
        times = np.vstack((zero_start_record, times))

    if course.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    r_dist = times[:, 0]
    r_secs = times[:, 1]
    c_dist = course[:, 0]

    if r_dist.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    c_secs = np.interp(c_dist, r_dist, r_secs)

    if times.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    max_time = times[-1, 1]
    if frequency_s <= 0:
        return np.empty((0, 3), dtype=np.float64)

    p_secs = np.arange(0.0, max_time + frequency_s, float(frequency_s))

    if p_secs.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    c_lon = course[:, 1]
    c_lat = course[:, 2]

    if c_secs.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    p_lon = np.interp(p_secs, c_secs, c_lon)
    p_lat = np.interp(p_secs, c_secs, c_lat)

    out_data = np.empty((p_secs.shape[0], 3), dtype=np.float64)
    out_data[:, 0] = p_secs
    out_data[:, 1] = p_lon
    out_data[:, 2] = p_lat

    return out_data
