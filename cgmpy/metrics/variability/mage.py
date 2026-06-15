"""MAGE (Mean Amplitude of Glycemic Excursions) metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────
# Pure functions
# ──────────────────────────────────────────────


def mage_simple(glucose: pd.Series, threshold: float | None = None) -> float:
    """Calculates MAGE (Mean Amplitude of Glycemic Excursions).

    Args:
        glucose: Glucose values.
        threshold: Minimum excursion magnitude to count.
            Uses glucose.std() if not provided.

    Returns:
        MAGE value.
    """
    if threshold is None:
        threshold = float(glucose.std())

    peaks_and_nadirs = pd.DataFrame(
        {
            "glucose": glucose[
                (glucose.shift(1) < glucose) & (glucose > glucose.shift(-1))
                | (glucose.shift(1) > glucose) & (glucose < glucose.shift(-1))
            ]
        }
    ).reset_index(drop=True)

    if len(peaks_and_nadirs) < 2:
        return 0.0

    excursions = []
    starts_with_peak = peaks_and_nadirs["glucose"][0] > peaks_and_nadirs["glucose"][1]

    for i in range(0, len(peaks_and_nadirs) - 1, 2):
        if starts_with_peak:
            peak, nadir = (
                peaks_and_nadirs["glucose"][i],
                peaks_and_nadirs["glucose"][i + 1],
            )
        else:
            nadir, peak = (
                peaks_and_nadirs["glucose"][i],
                peaks_and_nadirs["glucose"][i + 1],
            )
        if abs(peak - nadir) > threshold:
            excursions.append(abs(peak - nadir))

    return float(np.mean(excursions)) if excursions else 0.0


def mage_baghurst_smoothing(glucose: pd.Series, threshold: float | None = None) -> dict:
    """MAGE Baghurst approach 1: smoothing per original Baghurst algorithm.

    Args:
        glucose: Glucose values.
        threshold: Minimum excursion magnitude to count.
            Uses glucose.std() if not provided.

    Returns:
        Dictionary with MAGE+, MAGE-, MAGE_avg and related metrics.
    """
    vals = glucose.values
    sd = float(glucose.std())
    if threshold is None:
        threshold = sd

    # Guard: MAGE_Baghurst's smoothing uses a 9-point window centred on
    # each point (Baghurst 2007). With less than 9 points the convolution
    # produces edge artifacts and the algorithm is undefined. Return a
    # zeroed result rather than crash.
    # Also short-circuit constant data (sd == 0 -> threshold == 0 -> every
    # pair of equal points is a false excursion of magnitude 0).
    if len(vals) < 9 or sd == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
            "turning_points": [],
        }

    # Approach 1: Smoothing per original Baghurst algorithm
    turning_points = []

    # STEP 1: Apply smoothing filter and identify turning points in smoothed data
    weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46

    # Use np.convolve for central smoothing (much faster)
    smoothed = np.convolve(vals, weights, mode="same")

    # Adjust edges that np.convolve does not handle like the original Baghurst algorithm
    for i in range(min(4, len(vals))):
        smoothed[i] = vals[: i + 5].mean()
        if len(vals) > i + 5:
            smoothed[-(i + 1)] = vals[-(i + 5) :].mean()

    # Identify turning points in smoothed data via first differences
    delta = np.diff(smoothed)
    turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1

    # STEP 2: Identify local maxima/minima in original data
    for i in range(len(turning_smoothed) - 1):
        start = turning_smoothed[i]
        end = turning_smoothed[i + 1]

        # Find actual maximum in ascending interval
        if smoothed[start] < smoothed[end]:
            true_peak = np.argmax(vals[start:end]) + start
            turning_points.append(true_peak)
        # Find actual minimum in descending interval
        else:
            true_valley = np.argmin(vals[start:end]) + start
            turning_points.append(true_valley)

    # Add first and last point if they are extremes
    if (
        len(turning_points) > 0
        and turning_points[0] > 0
        and (vals[0] > vals[turning_points[0]] or vals[0] < vals[turning_points[0]])
    ):
        turning_points.insert(0, 0)

    if (
        len(turning_points) > 0
        and turning_points[-1] < len(vals) - 1
        and (vals[-1] > vals[turning_points[-1]] or vals[-1] < vals[turning_points[-1]])
    ):
        turning_points.append(len(vals) - 1)

    # STEP 3: Remove turning points associated with non-countable excursions on both sides
    # Keep those whose adjacent maxima/minima are lower/higher on both sides
    keep_iterating = True
    while keep_iterating:
        to_delete = []

        for i in range(1, len(turning_points) - 1):
            current_idx = turning_points[i]
            prev_idx = turning_points[i - 1]
            next_idx = turning_points[i + 1]

            current_val = vals[current_idx]
            prev_val = vals[prev_idx]
            next_val = vals[next_idx]

            # Check if both differences are below the threshold
            if abs(current_val - prev_val) < threshold and abs(current_val - next_val) < threshold:
                # Retain if it is a local max (both adjacent are lower)
                is_local_max = current_val > prev_val and current_val > next_val
                # Retain if it is a local min (both adjacent are higher)
                is_local_min = current_val < prev_val and current_val < next_val

                # If not a local max/min, mark for removal
                if not (is_local_max or is_local_min):
                    to_delete.append(i)

        # If no more points to remove, stop
        if not to_delete:
            keep_iterating = False
        else:
            # Remove marked points
            for idx in sorted(to_delete, reverse=True):
                turning_points.pop(idx)

        # STEP 4: Remove observations that are no longer turning points
        delta_turning = np.diff([vals[tp] for tp in turning_points])
        false_turning = []

        for i in range(1, len(delta_turning)):
            # If differences share the same sign, it is not a turning point
            if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                false_turning.append(i)

        # Remove false points
        for idx in sorted(false_turning, reverse=True):
            turning_points.pop(idx)

    # STEP 5: Remove turning points with a countable excursion on only one side
    if len(turning_points) >= 3:
        to_delete = []

        for i in range(1, len(turning_points) - 1):
            current_idx = turning_points[i]
            prev_idx = turning_points[i - 1]
            next_idx = turning_points[i + 1]

            current_val = vals[current_idx]
            prev_val = vals[prev_idx]
            next_val = vals[next_idx]

            # Check whether there is a significant excursion on only one side
            has_sig_prev = abs(current_val - prev_val) >= threshold
            has_sig_next = abs(current_val - next_val) >= threshold

            if has_sig_prev != has_sig_next:  # logical XOR - only one is true
                to_delete.append(i)

        # Remove marked points
        for idx in sorted(to_delete, reverse=True):
            turning_points.pop(idx)

        # Re-check for points that are no longer turning points
        delta_turning = np.diff([vals[tp] for tp in turning_points])
        false_turning = []

        for i in range(1, len(delta_turning)):
            if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                false_turning.append(i)

        for idx in sorted(false_turning, reverse=True):
            turning_points.pop(idx)

    # STEP 6: Remove non-countable excursions at the start or end
    if len(turning_points) >= 2:
        # Verify initial excursion
        if abs(vals[turning_points[0]] - vals[turning_points[1]]) < threshold:
            turning_points.pop(0)

        # Verify final excursion
        if (
            len(turning_points) >= 2
            and abs(vals[turning_points[-1]] - vals[turning_points[-2]]) < threshold
        ):
            turning_points.pop(-1)

    # Ensure points are sorted and unique
    turning_points = sorted(set(turning_points))

    # Calculate valid excursions
    excursions = []

    if len(turning_points) == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
        }

    last_val = vals[turning_points[0]]
    last_point = turning_points[0]

    for point in turning_points[1:]:
        curr_val = vals[point]
        diff = abs(curr_val - last_val)

        if diff >= threshold:
            excursions.append(
                {
                    "start_point": last_point,
                    "end_point": point,
                    "start": last_val,
                    "end": curr_val,
                    "type": "up" if curr_val > last_val else "down",
                    "magnitude": diff,
                }
            )
            last_val = curr_val
            last_point = point
        else:
            # If it does not exceed the threshold, update the last value without creating an excursion
            last_val = curr_val
            last_point = point

    # Separate excursions and compute metrics
    excursions_up = [e["magnitude"] for e in excursions if e["type"] == "up"]
    excursions_down = [e["magnitude"] for e in excursions if e["type"] == "down"]

    mage_plus = np.mean(excursions_up) if excursions_up else 0
    mage_minus = np.mean(excursions_down) if excursions_down else 0
    mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0

    return {
        "MAGE+": round(mage_plus, 2),
        "MAGE-": round(mage_minus, 2),
        "MAGE_avg": round(mage_avg, 2),
        "SD_used": round(sd, 2),
        "threshold": round(float(threshold), 2),
        "num_excursions": len(excursions),
        "turning_points": turning_points,
    }


def mage_baghurst_direct_elimination(glucose: pd.Series, threshold: float | None = None) -> dict:
    """MAGE Baghurst approach 2: direct elimination of intermediate points.

    Args:
        glucose: Glucose values.
        threshold: Minimum excursion magnitude to count.
            Uses glucose.std() if not provided.

    Returns:
        Dictionary with MAGE+, MAGE-, MAGE_avg and related metrics.
    """
    vals = glucose.values
    sd = float(glucose.std())
    if threshold is None:
        threshold = sd

    if len(vals) < 9 or sd == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
            "turning_points": [],
        }

    # Approach 2: Direct elimination
    turning_points = []
    i = 0

    # 1. First pass: remove intermediate points in monotonic sequences
    while i < len(vals) - 2:
        if (vals[i] <= vals[i + 1] <= vals[i + 2]) or (vals[i] >= vals[i + 1] >= vals[i + 2]):
            # The intermediate point is part of a monotonic sequence
            i += 1
        else:
            # Point i+1 is a potential turning point
            turning_points.append(i + 1)
            i += 2

    # Ensure we include the first and last point if relevant
    if len(turning_points) == 0 or turning_points[0] > 0:
        turning_points.insert(0, 0)
    if turning_points[-1] < len(vals) - 1:
        turning_points.append(len(vals) - 1)

    # 2. Second pass: remove excursions that do not exceed the threshold
    valid_points = []
    for i in range(1, len(turning_points) - 1):
        prev_val = vals[turning_points[i - 1]]
        curr_val = vals[turning_points[i]]
        next_val = vals[turning_points[i + 1]]

        # Check whether it is a valid max or min
        if (
            (curr_val > prev_val and curr_val > next_val)
            or (curr_val < prev_val and curr_val < next_val)
        ) and (abs(curr_val - prev_val) >= threshold or abs(curr_val - next_val) >= threshold):
            valid_points.append(turning_points[i])

    # Ensure we keep first and last points if needed
    if valid_points and valid_points[0] > 0:
        valid_points.insert(0, 0)
    if valid_points and valid_points[-1] < len(vals) - 1:
        valid_points.append(len(vals) - 1)

    turning_points = valid_points

    # Calculate valid excursions
    excursions = []

    if len(turning_points) == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
            "turning_points": [],
        }

    last_val = vals[turning_points[0]]
    last_point = turning_points[0]

    for point in turning_points[1:]:
        curr_val = vals[point]
        diff = abs(curr_val - last_val)

        if diff >= threshold:
            excursions.append(
                {
                    "start_point": last_point,
                    "end_point": point,
                    "start": last_val,
                    "end": curr_val,
                    "type": "up" if curr_val > last_val else "down",
                    "magnitude": diff,
                }
            )
            last_val = curr_val
            last_point = point
        else:
            last_val = curr_val
            last_point = point

    excursions_up = [e["magnitude"] for e in excursions if e["type"] == "up"]
    excursions_down = [e["magnitude"] for e in excursions if e["type"] == "down"]

    mage_plus = np.mean(excursions_up) if excursions_up else 0
    mage_minus = np.mean(excursions_down) if excursions_down else 0
    mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0

    return {
        "MAGE+": round(mage_plus, 2),
        "MAGE-": round(mage_minus, 2),
        "MAGE_avg": round(mage_avg, 2),
        "SD_used": round(sd, 2),
        "threshold": round(float(threshold), 2),
        "num_excursions": len(excursions),
        "turning_points": turning_points,
    }


def mage_baghurst_simplified(glucose: pd.Series, threshold: float | None = None) -> dict:
    """MAGE Baghurst approach 3: improved smoothing with additional filtering.

    Args:
        glucose: Glucose values.
        threshold: Minimum excursion magnitude to count.
            Uses glucose.std() if not provided.

    Returns:
        Dictionary with MAGE+, MAGE-, MAGE_avg and related metrics.
    """
    vals = glucose.values
    sd = float(glucose.std())
    if threshold is None:
        threshold = sd

    if len(vals) < 9 or sd == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
            "turning_points": [],
        }

    # Approach 3: Improved smoothing
    # 1. Apply smoothing filter with edge handling (same as approach 1)
    weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46
    smoothed = np.zeros_like(vals)

    # Central smoothing
    for i in range(4, len(vals) - 4):
        smoothed[i] = np.dot(weights, vals[i - 4 : i + 5])

    # Edge handling with simple mean
    for i in range(4):
        smoothed[i] = vals[: i + 5].mean()
        smoothed[-(i + 1)] = vals[-(i + 5) :].mean()

    # 2. Identify minima/maxima in the smoothed profile
    delta = np.diff(smoothed)
    turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1

    # 3. Find real turning points in original data between smoothed intervals
    # and apply additional filtering

    # First, identify all potential turning points
    potential_turning_points = []
    for i in range(len(turning_smoothed) - 1):
        start = turning_smoothed[i]
        end = turning_smoothed[i + 1]

        # Find actual maximum in ascending interval
        if smoothed[start] < smoothed[end]:
            true_peak = np.argmax(vals[start:end]) + start
            potential_turning_points.append((true_peak, "peak", vals[true_peak]))
        # Find actual minimum in descending interval
        else:
            true_valley = np.argmin(vals[start:end]) + start
            potential_turning_points.append((true_valley, "valley", vals[true_valley]))

    # Now process the turning points to drop minor intermediate peaks/valleys
    turning_points = []
    if potential_turning_points:
        # Add the first point
        turning_points.append(potential_turning_points[0][0])

        # Process the remaining points
        for i in range(1, len(potential_turning_points) - 1):
            prev_point, prev_type, prev_value = potential_turning_points[i - 1]
            curr_point, curr_type, curr_value = potential_turning_points[i]
            next_point, next_type, next_value = potential_turning_points[i + 1]

            # If we have a valley-peak-valley or peak-valley-peak pattern
            if curr_type == prev_type:
                # Skip this point, it is redundant
                continue

            # If we have a peak between two valleys, check if it is significant
            if (
                curr_type == "peak"
                and prev_type == "valley"
                and next_type == "valley"
                and (
                    curr_value - prev_value < threshold / 2
                    or curr_value - next_value < threshold / 2
                )
            ):
                # If the peak is not significantly higher than both valleys, skip it
                continue

            # If we have a valley between two peaks, check if it is significant
            if (
                curr_type == "valley"
                and prev_type == "peak"
                and next_type == "peak"
                and (
                    prev_value - curr_value < threshold / 2
                    or next_value - curr_value < threshold / 2
                )
            ):
                continue

            # If we reach here, the point is significant
            turning_points.append(curr_point)

        # Add the last point
        turning_points.append(potential_turning_points[-1][0])

    # Make sure we have at least the first and last point
    if len(turning_points) == 0 and len(vals) > 0:
        turning_points = [0, len(vals) - 1]
    elif len(turning_points) == 1 and len(vals) > 1:
        if turning_points[0] == 0:
            turning_points.append(len(vals) - 1)
        else:
            turning_points.insert(0, 0)

    turning_points = np.unique(turning_points)

    # Calculate valid excursions
    excursions = []

    if len(turning_points) == 0:
        return {
            "MAGE+": 0.0,
            "MAGE-": 0.0,
            "MAGE_avg": 0.0,
            "SD_used": round(sd, 2),
            "threshold": round(float(threshold), 2),
            "num_excursions": 0,
            "turning_points": [],
        }

    last_val = vals[int(turning_points[0])]
    last_point = int(turning_points[0])

    for point in turning_points[1:]:
        point = int(point)
        curr_val = vals[point]
        diff = abs(curr_val - last_val)

        if diff >= threshold:
            excursions.append(
                {
                    "start_point": last_point,
                    "end_point": point,
                    "start": last_val,
                    "end": curr_val,
                    "type": "up" if curr_val > last_val else "down",
                    "magnitude": diff,
                }
            )
            last_val = curr_val
            last_point = point
        else:
            last_val = curr_val
            last_point = point

    excursions_up = [e["magnitude"] for e in excursions if e["type"] == "up"]
    excursions_down = [e["magnitude"] for e in excursions if e["type"] == "down"]

    mage_plus = np.mean(excursions_up) if excursions_up else 0
    mage_minus = np.mean(excursions_down) if excursions_down else 0
    mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0

    return {
        "MAGE+": round(mage_plus, 2),
        "MAGE-": round(mage_minus, 2),
        "MAGE_avg": round(mage_avg, 2),
        "SD_used": round(sd, 2),
        "threshold": round(float(threshold), 2),
        "num_excursions": len(excursions),
        "turning_points": [int(tp) for tp in turning_points],
    }


def mage_baghurst(glucose: pd.Series, threshold: float | None = None) -> dict:
    """MAGE Baghurst: runs all 3 approaches and returns the best result.

    The "best" approach is the one with the most valid excursions.

    Args:
        glucose: Glucose values.
        threshold: Minimum excursion magnitude to count.
            Uses glucose.std() if not provided.

    Returns:
        Dictionary with MAGE+, MAGE-, MAGE_avg and related metrics
        from the best approach.
    """
    results = [
        mage_baghurst_smoothing(glucose, threshold),
        mage_baghurst_direct_elimination(glucose, threshold),
        mage_baghurst_simplified(glucose, threshold),
    ]
    best = max(results, key=lambda r: r["num_excursions"])
    return best
