"""MAGE (Mean Amplitude of Glycemic Excursions) metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import VariabilityBase

if TYPE_CHECKING:
    pass


class MAGEMetrics(VariabilityBase):
    """Mixin providing MAGE calculations for glucose data.

    All methods assume the host class also provides the
    :class:`~cgmpy.data.core.ModularGlucoseData` interface (``data``,
    ``sd``, ``mean``, ``typical_interval``, etc.).
    """

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float

        def sd(self) -> float: ...
        def mean(self) -> float: ...

    def MAGE_Baghurst(self, threshold_sd: int = 1, approach: int = 1, plot: bool = False) -> dict:
        """
        Calculates MAGE using the specific algorithm from Baghurst.

        Main changes:
        1. Correct handling of edges in smoothing
        2. Search for turning points in original data between minima/maxima of smoothed profile
        3. Iterative process of eliminating invalid points
        4. Handling of excursions at the beginning/end of the dataset

        :param threshold_sd: Number of standard deviations for the threshold
        :param approach: 1 to use smoothing per original Baghurst,
                        2 for direct elimination, 3 for improved smoothing
        :param plot: If True, generates a visualization of identified peaks and valleys
        :return: Dictionary with MAGE+, MAGE- and related metrics

        Approach 1: Original Baghurst algorithm with smoothing and iterative elimination process
        Approach 2: Direct elimination of intermediate points in monotonic sequences
        Approach 3: Improved smoothing with additional turning point filtering
        """
        glucose = self.data["glucose"].values
        times = self.data["time"].values
        sd = self.sd()
        threshold = threshold_sd * sd

        # Guard: MAGE_Baghurst's smoothing uses a 9-point window centred on
        # each point (Baghurst 2007). With less than 9 points the convolution
        # produces edge artifacts and the algorithm is undefined. Return a
        # zeroed result rather than crash.
        # Also short-circuit constant data (sd == 0 → threshold == 0 → every
        # pair of equal points is a false excursion of magnitude 0).
        if len(glucose) < 9 or sd == 0:
            return {
                "MAGE+": 0.0,
                "MAGE-": 0.0,
                "MAGE_avg": 0.0,
                "SD_used": round(float(sd), 2),
                "threshold": round(float(threshold), 2),
                "num_excursions": 0,
            }

        # Store turning points for each approach if plot=True
        turning_points_approaches = {}

        # Approach 1: Smoothing per original Baghurst algorithm
        if approach == 1 or plot:
            # STEP 1: Apply smoothing filter and identify turning points in smoothed data
            weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46

            # Use np.convolve for central smoothing (much faster)
            smoothed = np.convolve(glucose, weights, mode="same")

            # Adjust edges that np.convolve does not handle like the original Baghurst algorithm
            for i in range(min(4, len(glucose))):
                smoothed[i] = glucose[: i + 5].mean()
                if len(glucose) > i + 5:
                    smoothed[-(i + 1)] = glucose[-(i + 5) :].mean()

            # Identify turning points in smoothed data via first differences
            delta = np.diff(smoothed)
            turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1

            # STEP 2: Identify local maxima/minima in original data
            turning_points_1 = []
            for i in range(len(turning_smoothed) - 1):
                start = turning_smoothed[i]
                end = turning_smoothed[i + 1]

                # Find actual maximum in ascending interval
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    turning_points_1.append(true_peak)
                # Find actual minimum in descending interval
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    turning_points_1.append(true_valley)

            # Add first and last point if they are extremes
            if (
                len(turning_points_1) > 0
                and turning_points_1[0] > 0
                and (
                    glucose[0] > glucose[turning_points_1[0]]
                    or glucose[0] < glucose[turning_points_1[0]]
                )
            ):
                turning_points_1.insert(0, 0)

            if (
                len(turning_points_1) > 0
                and turning_points_1[-1] < len(glucose) - 1
                and (
                    glucose[-1] > glucose[turning_points_1[-1]]
                    or glucose[-1] < glucose[turning_points_1[-1]]
                )
            ):
                turning_points_1.append(len(glucose) - 1)

            # STEP 3: Remove turning points associated with non-countable excursions on both sides
            # Keep those whose adjacent maxima/minima are lower/higher on both sides
            keep_iterating = True
            while keep_iterating:
                to_delete = []

                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i - 1]
                    next_idx = turning_points_1[i + 1]

                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]

                    # Check if both differences are below the threshold
                    if (
                        abs(current_val - prev_val) < threshold
                        and abs(current_val - next_val) < threshold
                    ):
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
                        turning_points_1.pop(idx)

                # STEP 4: Remove observations that are no longer turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []

                for i in range(1, len(delta_turning)):
                    # If differences share the same sign, it is not a turning point
                    if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                        false_turning.append(i)

                # Remove false points
                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)

            # STEP 5: Remove turning points with a countable excursion on only one side
            if len(turning_points_1) >= 3:
                to_delete = []

                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i - 1]
                    next_idx = turning_points_1[i + 1]

                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]

                    # Check whether there is a significant excursion on only one side
                    has_sig_prev = abs(current_val - prev_val) >= threshold
                    has_sig_next = abs(current_val - next_val) >= threshold

                    if has_sig_prev != has_sig_next:  # logical XOR - only one is true
                        to_delete.append(i)

                # Remove marked points
                for idx in sorted(to_delete, reverse=True):
                    turning_points_1.pop(idx)

                # Re-check for points that are no longer turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []

                for i in range(1, len(delta_turning)):
                    if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                        false_turning.append(i)

                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)

            # STEP 6: Remove non-countable excursions at the start or end
            if len(turning_points_1) >= 2:
                # Verify initial excursion
                if abs(glucose[turning_points_1[0]] - glucose[turning_points_1[1]]) < threshold:
                    turning_points_1.pop(0)

                # Verify final excursion
                if (
                    len(turning_points_1) >= 2
                    and abs(glucose[turning_points_1[-1]] - glucose[turning_points_1[-2]])
                    < threshold
                ):
                    turning_points_1.pop(-1)

            # Ensure points are sorted and unique
            turning_points_1 = sorted(set(turning_points_1))
            turning_points_approaches[1] = turning_points_1

            if approach == 1:
                turning_points = turning_points_1

        # Approach 2: Direct elimination
        if approach == 2 or plot:
            turning_points_2 = []
            i = 0

            # 1. First pass: remove intermediate points in monotonic sequences
            while i < len(glucose) - 2:
                if (glucose[i] <= glucose[i + 1] <= glucose[i + 2]) or (
                    glucose[i] >= glucose[i + 1] >= glucose[i + 2]
                ):
                    # The intermediate point is part of a monotonic sequence
                    i += 1
                else:
                    # Point i+1 is a potential turning point
                    turning_points_2.append(i + 1)
                    i += 2

            # Ensure we include the first and last point if relevant
            if len(turning_points_2) == 0 or turning_points_2[0] > 0:
                turning_points_2.insert(0, 0)
            if turning_points_2[-1] < len(glucose) - 1:
                turning_points_2.append(len(glucose) - 1)

            # 2. Second pass: remove excursions that do not exceed the threshold
            valid_points = []
            for i in range(1, len(turning_points_2) - 1):
                prev_val = glucose[turning_points_2[i - 1]]
                curr_val = glucose[turning_points_2[i]]
                next_val = glucose[turning_points_2[i + 1]]

                # Check whether it is a valid max or min
                if (
                    (curr_val > prev_val and curr_val > next_val)
                    or (curr_val < prev_val and curr_val < next_val)
                ) and (
                    abs(curr_val - prev_val) >= threshold or abs(curr_val - next_val) >= threshold
                ):
                    valid_points.append(turning_points_2[i])

            # Ensure we keep first and last points if needed
            if valid_points and valid_points[0] > 0:
                valid_points.insert(0, 0)
            if valid_points and valid_points[-1] < len(glucose) - 1:
                valid_points.append(len(glucose) - 1)

            turning_points_2 = valid_points
            turning_points_approaches[2] = turning_points_2

            if approach == 2:
                turning_points = turning_points_2

        # Approach 3: Improved smoothing
        if approach == 3 or plot:
            # 1. Apply smoothing filter with edge handling (same as approach 1)
            weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46
            smoothed = np.zeros_like(glucose)

            # Central smoothing
            for i in range(4, len(glucose) - 4):
                smoothed[i] = np.dot(weights, glucose[i - 4 : i + 5])

            # Edge handling with simple mean
            for i in range(4):
                smoothed[i] = glucose[: i + 5].mean()
                smoothed[-(i + 1)] = glucose[-(i + 5) :].mean()

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
                    true_peak = np.argmax(glucose[start:end]) + start
                    potential_turning_points.append((true_peak, "peak", glucose[true_peak]))
                # Find actual minimum in descending interval
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    potential_turning_points.append((true_valley, "valley", glucose[true_valley]))

            # Now process the turning points to drop minor intermediate peaks/valleys
            turning_points_3 = []
            if potential_turning_points:
                # Add the first point
                turning_points_3.append(potential_turning_points[0][0])

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
                    turning_points_3.append(curr_point)

                # Add the last point
                turning_points_3.append(potential_turning_points[-1][0])

            # Make sure we have at least the first and last point
            if len(turning_points_3) == 0 and len(glucose) > 0:
                turning_points_3 = [0, len(glucose) - 1]
            elif len(turning_points_3) == 1 and len(glucose) > 1:
                if turning_points_3[0] == 0:
                    turning_points_3.append(len(glucose) - 1)
                else:
                    turning_points_3.insert(0, 0)

            turning_points_3 = np.unique(turning_points_3)
            turning_points_approaches[3] = turning_points_3

            if approach == 3:
                turning_points = turning_points_3

        # 3. Calculate valid excursions
        excursions = []

        # Guard: if no turning points survived (too few points or all excursions
        # were below threshold), return a zeroed result with the same shape.
        if turning_points is None or len(turning_points) == 0:
            return {
                "MAGE+": 0.0,
                "MAGE-": 0.0,
                "MAGE_avg": 0.0,
                "SD_used": round(sd, 2),
                "threshold": round(threshold, 2),
                "num_excursions": 0,
            }

        last_val = glucose[turning_points[0]]
        last_point = turning_points[0]

        for point in turning_points[1:]:
            curr_val = glucose[point]
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
        mage_avg = (
            np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0
        )

        # Generate visualization if plot=True
        if plot:
            from datetime import timedelta

            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            # Get all unique days in the data
            unique_days = pd.Series(times).dt.normalize().unique()

            # Configure figure and axes - now with 3 subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            plt.ion()  # Interactive mode

            # Compute excursions for each approach
            excursions_by_approach = {}

            for approach_num in [1, 2, 3]:
                # Use the specific turning points of this approach
                if approach_num in turning_points_approaches:
                    tp = turning_points_approaches[approach_num]

                    # Compute excursions for this approach
                    excursions_approach = []
                    if len(tp) > 1:
                        last_val = glucose[tp[0]]
                        last_point = tp[0]

                        for point in tp[1:]:
                            curr_val = glucose[point]
                            diff = abs(curr_val - last_val)

                            if diff >= threshold:
                                excursions_approach.append(
                                    {
                                        "start_point": last_point,
                                        "end_point": point,
                                        "start": last_val,
                                        "end": curr_val,
                                        "type": "up" if curr_val > last_val else "down",
                                        "magnitude": diff,
                                    }
                                )

                            # Always update the last value and point
                            last_val = curr_val
                            last_point = point

                    excursions_by_approach[approach_num] = excursions_approach

            # Function to update the plot for a specific day
            def update_plot(day_index):
                # Clear the axes
                for ax in axs:
                    ax.clear()

                # Get the current day
                current_day = unique_days[day_index]
                next_day = current_day + timedelta(days=1)

                # Filter data to show only the current day
                day_mask = (times >= current_day) & (times < next_day)
                day_times = times[day_mask]
                day_glucose = glucose[day_mask]

                if len(day_times) > 0:
                    # For each approach
                    for i, approach in enumerate([1, 2, 3]):
                        ax = axs[i]
                        # Draw the glucose line
                        ax.plot(day_times, day_glucose, "b-", label="Glucosa")

                        # Get turning points for this approach
                        approach_turning_points = turning_points_approaches.get(approach, [])

                        # Get excursions for this approach
                        approach_excursions = excursions_by_approach.get(approach, [])

                        # Filter turning points for this day
                        day_turning_points = [tp for tp in approach_turning_points if day_mask[tp]]

                        # Identify points involved in excursions
                        excursion_points = set()
                        day_excursions = []

                        for exc in approach_excursions:
                            start_point = exc["start_point"]
                            end_point = exc["end_point"]

                            # Check if the excursion belongs to the current day
                            if day_mask[start_point] or day_mask[end_point]:
                                if day_mask[start_point]:
                                    excursion_points.add(start_point)
                                if day_mask[end_point]:
                                    excursion_points.add(end_point)
                                day_excursions.append(exc)

                        # Classify turning points
                        significant_points = [
                            tp for tp in day_turning_points if tp in excursion_points
                        ]
                        non_significant_points = [
                            tp for tp in day_turning_points if tp not in excursion_points
                        ]

                        # Draw non-significant points in blue
                        for tp in non_significant_points:
                            ax.plot(times[tp], glucose[tp], "bo", markersize=6)

                        # Draw significant points in red
                        for tp in significant_points:
                            ax.plot(times[tp], glucose[tp], "ro", markersize=8)

                        # Draw lines for the excursions
                        for exc in day_excursions:
                            start_point = exc["start_point"]
                            end_point = exc["end_point"]

                            # Ensure both points belong to the current day
                            if day_mask[start_point] and day_mask[end_point]:
                                # Draw a thick line whose color depends on the excursion type
                                color = "green" if exc["type"] == "up" else "red"
                                ax.plot(
                                    [times[start_point], times[end_point]],
                                    [glucose[start_point], glucose[end_point]],
                                    color=color,
                                    linewidth=2.5,
                                    alpha=0.7,
                                )

                        # Compute MAGE for this approach and day
                        excursions_up = [
                            e["magnitude"] for e in day_excursions if e["type"] == "up"
                        ]
                        excursions_down = [
                            e["magnitude"] for e in day_excursions if e["type"] == "down"
                        ]

                        mage_plus = np.mean(excursions_up) if excursions_up else 0
                        mage_minus = np.mean(excursions_down) if excursions_down else 0
                        mage_avg = (
                            np.mean(excursions_up + excursions_down)
                            if (excursions_up or excursions_down)
                            else 0
                        )

                        # Configure title and labels
                        approach_name = (
                            "Suavizado"
                            if approach == 1
                            else "Direct elimination"
                            if approach == 2
                            else "Improved smoothing"
                        )
                        ax.set_title(
                            f"MAGE Baghurst - Approach {approach} ({approach_name}) - "
                            f"{current_day.strftime('%d/%m/%Y')}\n"
                            f"MAGE+: {mage_plus:.1f}, MAGE-: {mage_minus:.1f}, "
                            f"MAGE: {mage_avg:.1f}, Excursions: {len(day_excursions)}"
                        )
                        ax.set_ylabel("Glucose (mg/dL)")
                        ax.grid(True)
                        ax.axhline(
                            y=self.mean() + threshold,
                            color="g",
                            linestyle="--",
                            label=f"Threshold (+{threshold_sd} SD)",
                        )
                        ax.axhline(
                            y=self.mean() - threshold,
                            color="g",
                            linestyle="--",
                            label=f"Threshold (-{threshold_sd} SD)",
                        )

                        # Custom legend
                        from matplotlib.lines import Line2D

                        custom_lines = [
                            Line2D(
                                [0],
                                [0],
                                color="b",
                                marker="o",
                                linestyle="None",
                                markersize=6,
                            ),
                            Line2D(
                                [0],
                                [0],
                                color="r",
                                marker="o",
                                linestyle="None",
                                markersize=8,
                            ),
                            Line2D([0], [0], color="green", linewidth=2.5),
                            Line2D([0], [0], color="red", linewidth=2.5),
                            Line2D([0], [0], color="g", linestyle="--"),
                        ]
                        ax.legend(
                            custom_lines,
                            [
                                "Turning points",
                                "Excursion points",
                                "Positive excursion",
                                "Negative excursion",
                                "Threshold (±1 SD)",
                            ],
                        )

                    # Configure x-axis for the last plot
                    axs[2].set_xlabel("Time")

                    # Format x-axis to display hours
                    for ax in axs:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                    # Add navigation information
                    fig.suptitle(
                        f"Day {day_index + 1} of {len(unique_days)} - Press left/right arrows to navigate, q to quit",
                        fontsize=12,
                    )

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)  # Make room for the top title
                    fig.canvas.draw_idle()

            # Index of the current day
            current_day_index = 0

            # Function to handle keyboard events
            def on_key(event):
                nonlocal current_day_index

                if event.key == "right" and current_day_index < len(unique_days) - 1:
                    current_day_index += 1
                    update_plot(current_day_index)
                elif event.key == "left" and current_day_index > 0:
                    current_day_index -= 1
                    update_plot(current_day_index)
                elif event.key == "q":
                    plt.close(fig)

            # Connect the keyboard event
            fig.canvas.mpl_connect("key_press_event", on_key)

            # Show the first day
            update_plot(current_day_index)

            # Show instructions
            self.logger.info("Navigation:")
            self.logger.info("  <- Left arrow: Previous day")
            self.logger.info("  -> Right arrow: Next day")
            self.logger.info("  q: Quit")

            # Block until the figure is closed
            plt.show(block=True)

        return {
            "MAGE+": round(mage_plus, 2),
            "MAGE-": round(mage_minus, 2),
            "MAGE_avg": round(mage_avg, 2),
            "SD_used": round(sd, 2),
            "threshold": round(threshold, 2),
            "num_excursions": len(excursions),
        }

    def MAGE(self) -> float:
        """
        Calculates MAGE (Mean Amplitude of Glycemic Excursions).
        :return: MAGE value.
        """
        sd = self.sd()
        peaks_and_nadirs = self.data[
            (self.data["glucose"].shift(1) < self.data["glucose"])
            & (self.data["glucose"] > self.data["glucose"].shift(-1))
            | (self.data["glucose"].shift(1) > self.data["glucose"])
            & (self.data["glucose"] < self.data["glucose"].shift(-1))
        ].reset_index(drop=True)

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
            if abs(peak - nadir) > sd:
                excursions.append(abs(peak - nadir))

        return float(np.mean(excursions)) if excursions else 0.0
