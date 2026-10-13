"""Approximate 2D prediction, NOT a 3D landing or court line-call system."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Landing:
    x: float
    y: float
    seconds: float
    path: list
    held: bool = False
    provisional: bool = False
    model: str = "image"
    floor_point: tuple | None = None
    uncertainty_m: float | None = None
    uncertainty_px: float | None = None
    retained: bool = False
    first_locked: bool = False
    observation_count: int = 0
    observation_span: float = 0.0
    motion_model: str = "flight"


def _motion_fit(values):
    """Fit a short arc robustly, then estimate coefficient noise sensitivity."""
    times = values[:, 0]-values[-1, 0]
    span = float(np.ptp(times))
    scaled = times/span
    x_design = np.column_stack([scaled, np.ones(len(scaled))])
    y_design = np.column_stack([scaled**2, scaled, np.ones(len(scaled))])
    weights = np.ones(len(values))
    for _ in range(6):
        root_weight = np.sqrt(weights)
        x_fit = np.linalg.lstsq(x_design*root_weight[:, None],
                               values[:, 1]*root_weight, rcond=None)[0]
        y_fit = np.linalg.lstsq(y_design*root_weight[:, None],
                               values[:, 2]*root_weight, rcond=None)[0]
        errors = np.hypot(x_design@x_fit-values[:, 1], y_design@y_fit-values[:, 2])
        # A single wrong center should not bend the entire future arc.
        threshold = max(3.0, 2*np.median(errors))
        weights = np.minimum(1, threshold/np.maximum(errors, 1e-9))
    inliers = errors <= max(6.0, 2.5*np.median(errors))
    if not inliers[-1] and not inliers[-2]:
        return None  # New motion must not be discarded as repeated "outliers".
    if np.count_nonzero(inliers) < max(5, int(np.ceil(len(values)*0.75))):
        return None
    clean = values[inliers]
    if np.ptp(clean[:, 0]) < 0.12 or np.ptp(clean[:, 2]) < 3:
        return None
    x_design, y_design = x_design[inliers], y_design[inliers]
    x_fit = np.linalg.lstsq(x_design, clean[:, 1], rcond=None)[0]
    y_fit = np.linalg.lstsq(y_design, clean[:, 2], rcond=None)[0]
    residual = np.column_stack([x_design@x_fit-clean[:, 1], y_design@y_fit-clean[:, 2]])
    variance = max(1.5**2, float(np.sum(residual**2))/max(1, 2*len(clean)-5))
    quad_cov = np.linalg.inv(y_design.T@y_design)*variance
    curvature_significant = abs(y_fit[0]) > 2*np.sqrt(quad_cov[0, 0])
    linear_fit = np.linalg.lstsq(x_design, clean[:, 2], rcond=None)[0]
    linear_rmse = float(np.sqrt(np.mean((x_design@linear_fit-clean[:, 2])**2)))
    linear = not curvature_significant and linear_rmse <= 2.5 and linear_fit[0]/span > 20
    covariance = np.zeros((5, 5))
    covariance[:2, :2] = np.linalg.inv(x_design.T@x_design)*variance
    if linear:
        y_fit = np.r_[0.0, linear_fit]
        covariance[3:, 3:] = np.linalg.inv(x_design.T@x_design)*variance
        residual[:, 1] = x_design@linear_fit-clean[:, 2]
    else:
        covariance[2:, 2:] = quad_cov
    # Convert normalized time coefficients/covariance back into seconds.
    scale = np.diag([1/span, 1, 1/span**2, 1/span, 1])
    covariance = scale@covariance@scale.T
    x_coeff = x_fit*np.array([1/span, 1])
    y_coeff = y_fit*np.array([1/span**2, 1/span, 1])
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return x_coeff, y_coeff, covariance, rmse, linear, len(clean), float(np.ptp(clean[:, 0]))


def predict_landing(samples, now, ground_y, width, horizon=1.5, max_rmse=12.0,
                    ground=None, height=None):
    """Use recent measured arcs to predict an approximate 2D line crossing."""
    if len(samples) < 5:
        return None, "Waiting: need 5 measured positions"
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        return None, "Waiting: invalid observations"
    if np.any(np.diff(values[:, 0]) <= 0):
        return None, "Waiting: invalid observation times"
    results = []
    reason = "Waiting: need a clean moving arc"
    # Shorter tails exclude holding the ball before release or an old motion
    # segment, while longer windows reduce extrapolation noise on clean flights.
    for count in sorted(set([len(samples), min(16, len(samples)), min(10, len(samples))]), reverse=True):
        landing, message = _predict_window(values[-count:], now, ground_y, width,
                                          horizon, max_rmse, ground, height)
        if landing is not None:
            score = landing.uncertainty_px + (8 if landing.provisional else 0)
            results.append((score, landing))
        elif count == len(samples):
            reason = message
    return (min(results, key=lambda item: item[0])[1], "Approximate 2D line crossing") if results else (None, reason)


def _predict_window(values, now, ground_y, width, horizon, max_rmse, ground, height):
    """Fit x linearly and y quadratically to recent (time, x, y) observations.

    Centering time at the last observation avoids poor numerical conditioning
    when a camera has been running for a long time. A future downward root is
    used because image Y grows downward. Return a reason when withholding a fit.
    """
    age = now - values[-1, 0]
    if age < 0 or age > 0.25:
        return None, "Waiting: observations are stale"
    t = values[:, 0] - values[-1, 0]
    if np.ptp(t) < 0.12:
        return None, "Waiting: need a longer observed arc"
    if np.ptp(values[:, 2]) < 3:
        return None, "Waiting: need visible vertical motion"

    fit = _motion_fit(values)
    if fit is None:
        return None, "Waiting: need a clean moving arc"
    x_coeff, y_coeff, covariance, rmse, early_linear, count, observed_span = fit
    if rmse > max_rmse:
        return None, "Waiting: trajectory fit is too noisy"
    a, b, c = y_coeff
    if a < -1e-6:
        return None, "Waiting: curve is not downward"
    # y_ground(x)=m*x+k follows the marked guide in the floor projection.
    # Substitute x(t)=vx*t+x0, so the intersection is still a quadratic.
    if ground is not None:
        if ground.mode is not None:
            return None, "Waiting: finish floor/contact calibration"
        if height is None:
            raise ValueError("Frame height is required with a floor map.")
        first, second = ground.endpoints(width, height)
        if second[0] - first[0] < 1:
            return None, "Waiting: contact guide is nearly vertical"
        ground_slope = (second[1] - first[1]) / (second[0] - first[0])
        ground_intercept = first[1] - ground_slope * first[0]
        min_x, max_x = first[0], second[0]
    else:
        ground_slope, ground_intercept = 0.0, float(ground_y)
        min_x, max_x = 0, width - 1
    relative_b = b - ground_slope * x_coeff[0]
    relative_c = c - ground_slope * x_coeff[1] - ground_intercept
    if values[-1, 2] >= ground_slope*values[-1, 1]+ground_intercept:
        return None, "Waiting: observed ball already crossed the line"
    if a * age**2 + relative_b * age + relative_c >= 0 or relative_c >= 0:
        return None, "Waiting: ball is already at/below the line"

    # Handle the near-linear case separately rather than dividing by tiny a.
    if abs(a) < 1e-6:
        roots = [-relative_c / relative_b] if relative_b > 1e-6 else []
    else:
        discriminant = relative_b**2 - 4 * a * relative_c
        roots = [] if discriminant < 0 else [
            (-relative_b - np.sqrt(discriminant)) / (2 * a),
            (-relative_b + np.sqrt(discriminant)) / (2 * a)]
    for root in sorted(roots):
        remaining = root - age
        allowed_horizon = min(horizon, 0.75) if early_linear else horizon
        if remaining <= 0 or remaining > allowed_horizon or 2 * a * root + relative_b <= 0:
            continue
        x = float(np.polyval(x_coeff, root))
        y = float(ground_slope * x + ground_intercept)
        if not 0 <= x < width or not min_x <= x <= max_x:
            continue
        if ground is not None and (not 0 <= y < height or not ground.contains(x, y)):
            continue
        # Implicitly differentiate the intersection equation to measure how
        # detector noise changes the crossing time and horizontal target.
        derivative = 2*a*root + relative_b
        time_gradient = -np.array([-ground_slope*root, -ground_slope,
                                  root**2, root, 1])/derivative
        x_gradient = np.array([root, 1, 0, 0, 0]) + x_coeff[0]*time_gradient
        uncertainty = float(2*np.sqrt(max(0, x_gradient@covariance@x_gradient)))
        time_uncertainty = float(2*np.sqrt(max(0, time_gradient@covariance@time_gradient)))
        if uncertainty > max(25, width*0.12) or time_uncertainty > max(0.15, remaining*0.75):
            return None, "Waiting: extrapolation is uncertain; show more flight"
        provisional = early_linear or count < 8 or np.ptp(t) < 0.25
        times = np.linspace(age, root, 24)
        path = list(zip(np.polyval(x_coeff, times), np.polyval(y_coeff, times)))
        return Landing(x, y, float(remaining), path, provisional=provisional,
                       uncertainty_px=uncertainty, observation_count=count,
                       observation_span=observed_span), "Approximate 2D line crossing"
    return None, "Waiting: no plausible future intersection"


def draw_prediction(frame, landing, reason, ground_y=None, ground=None):
    height, width = frame.shape[:2]
    if ground is not None:
        ground.draw(frame)
    elif ground_y is not None:
        cv2.line(frame, (0, ground_y), (width - 1, ground_y), (255, 190, 60), 1)
    blue = (255, 100, 0)  # OpenCV uses BLUE, GREEN, RED channel order.
    if landing is not None:
        for index, (first, second) in enumerate(zip(landing.path, landing.path[1:])):
            if index % 2 == 0:
                cv2.line(frame, tuple(map(round, first)), tuple(map(round, second)),
                         blue, 2)
        point = (round(landing.x), round(landing.y))
        # Dark outer stroke and concentric blue rings stay visible on bright floors.
        cv2.circle(frame, point, 21, (30, 30, 30), 5, cv2.LINE_AA)
        for radius in (21, 12):
            cv2.circle(frame, point, radius, blue, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, blue, -1, cv2.LINE_AA)
        if landing.model == "plane":
            prefix = ("Held plane estimate" if landing.held else
                      "Early plane estimate" if landing.provisional else "Experimental plane estimate")
            reason = f"{prefix}: {landing.seconds:.2f}s to floor"
            note = ("DROP MODEL: assumes zero sideways motion; box size supplies approximate depth"
                    if landing.motion_model == "drop" else
                    "Assumed lens + resting-ball size reference; accuracy unverified")
            cv2.putText(frame, note,
                        (10, height-38), cv2.FONT_HERSHEY_SIMPLEX, 0.43, blue, 1)
        elif landing.model == "physics":
            prefix = ("Held flight estimate" if landing.held else
                      "Early flight estimate" if landing.provisional else "Flight estimate")
            reason = f"{prefix}: {landing.seconds:.2f}s to floor"
            x_m, y_m = landing.floor_point
            cv2.putText(frame, f"Floor X={x_m:.2f}m Y={y_m:.2f}m | noise sensitivity ~{landing.uncertainty_m:.2f}m",
                        (10, height-38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)
        else:
            prefix = ("Held 2D crossing" if landing.held else
                      "Early 2D crossing" if landing.provisional else "2D line crossing")
            reason = f"{prefix}: {landing.seconds:.2f}s"
            if landing.uncertainty_px is not None:
                cv2.putText(frame, f"Fit spread ~+/-{landing.uncertainty_px:.0f}px | real landing depth unknown",
                            (10, height-38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)
        if landing.retained:
            reason = ("First prediction locked | R: next throw" if landing.first_locked
                      else "Saved prediction | R: clear for next throw")
    cv2.putText(frame, reason, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, blue, 2)
