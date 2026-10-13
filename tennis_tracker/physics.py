"""Experimental single-camera ballistic landing estimate.

Known lens calibration, a measured floor plane, timestamps, and gravity give
additional constraints that image-only curve fitting did not have. They do not
make arbitrary throws observable: nearly straight viewing rays remain ambiguous.
"""
import numpy as np

from tennis_tracker.prediction import Landing


GRAVITY = 9.81


def _fit(camera, rotation, samples):
    times = samples[:, 0] - samples[-1, 0]
    span = float(np.ptp(times))
    scaled_time = times / span
    rays = camera.normalized(samples[:, 1:])
    acceleration = rotation @ np.array([0.0, 0.0, -GRAVITY])
    gravity_offset = 0.5 * times[:, None]**2 * acceleration

    # q(t) = position + (velocity*span)*(t/span) + 0.5*gravity*t^2.
    # For each viewing ray (u,v,1), q_x-u*q_z=0 and q_y-v*q_z=0.
    # These equations are linear in six unknown position/velocity components.
    rows, targets = [], []
    for tau, (u, v), offset in zip(scaled_time, rays, gravity_offset):
        for direction in (np.array([1, 0, -u]), np.array([0, 1, -v])):
            rows.append(np.r_[direction, tau*direction])
            targets.append(-direction @ offset)
    design = np.asarray(rows)
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0]/singular[-1] > 10000:
        return None
    state = np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]

    def residual_and_jacobian(value):
        points = value[:3] + scaled_time[:, None]*value[3:] + gravity_offset
        if np.any(points[:, 2] < 0.1):
            return None
        z = points[:, 2]
        normalized = points[:, :2] / z[:, None]
        error = normalized - rays
        # Differentiate x/z and y/z, so distant/near points are weighted in
        # image coordinates rather than biased by their unknown camera distance.
        jacobian = np.zeros((len(points), 2, 6))
        jacobian[:, 0, 0] = 1/z
        jacobian[:, 1, 1] = 1/z
        jacobian[:, 0, 2] = -points[:, 0]/z**2
        jacobian[:, 1, 2] = -points[:, 1]/z**2
        jacobian[:, :, 3:] = jacobian[:, :, :3]*scaled_time[:, None, None]
        focal = np.array([camera.matrix[0, 0], camera.matrix[1, 1]])
        return error*focal, jacobian*focal[None, :, None], points

    # A few small Gauss-Newton refinements minimize pixel reprojection error.
    # Robust weights limit the influence of occasional detector-center outliers.
    for _ in range(8):
        result = residual_and_jacobian(state)
        if result is None:
            return None
        error, jacobian, _ = result
        weights = np.minimum(1.0, 4 / np.maximum(np.linalg.norm(error, axis=1), 1e-6))
        weighted_j = (jacobian*np.sqrt(weights)[:, None, None]).reshape(-1, 6)
        weighted_error = (error*np.sqrt(weights)[:, None]).ravel()
        change = np.linalg.lstsq(weighted_j, -weighted_error, rcond=None)[0]
        # Only accept steps that reduce the actual reprojection residual.
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.1):
            trial = residual_and_jacobian(state + factor*change)
            if trial is not None and np.sum(trial[0]**2) <= np.sum(error**2) + 1e-9:
                state += factor*change
                accepted = True
                break
        if not accepted or np.linalg.norm(change) < 1e-7:
            break
    error, jacobian, points = residual_and_jacobian(state)
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    # This measures observation-noise sensitivity, not all calibration/model error.
    flat_j = jacobian.reshape(-1, 6)
    noise_pixels = max(1.5, rmse)
    covariance = np.linalg.pinv(flat_j.T @ flat_j, rcond=1e-12)*noise_pixels**2
    return state, span, points, rmse, covariance


def _impact(state, span, floor, radius):
    position = floor.rotation.T @ (state[:3] - floor.offset)
    velocity = floor.rotation.T @ (state[3:]/span)
    height = position[2] - radius
    if height <= 0:
        return None
    time = (velocity[2] + np.sqrt(velocity[2]**2 + 2*GRAVITY*height))/GRAVITY
    point = position + velocity*time + np.array([0, 0, -0.5*GRAVITY*time**2])
    return point, float(time), position, velocity


def predict_physics(samples, now, floor, horizon=2.0, radius=0.0335,
                    max_uncertainty=0.75, motion_model="flight", size_observations=None):
    dropping = motion_model == "drop"
    minimum_points = 5 if dropping else 6
    minimum_span = 0.10 if dropping else 0.18
    if not floor.ready:
        return None, "Waiting: calibrate lens and measured floor first"
    if len(samples) < minimum_points:
        return None, f"Waiting: need {minimum_points} measured positions"
    observations = np.asarray(samples, float)
    if not np.isfinite(observations).all() or np.any(np.diff(observations[:, 0]) <= 0):
        return None, "Waiting: invalid observation times"
    age = now - observations[-1, 0]
    if not 0 <= age <= 0.25:
        return None, "Waiting: observations are stale"
    if np.ptp(observations[:, 0]) < minimum_span:
        return None, f"Waiting: need {minimum_span:.2f}s of visible motion"
    if np.linalg.norm(np.ptp(observations[:, 1:], axis=0)) < (6 if dropping else 12):
        return None, "Waiting: ball is stationary or motion is too small"

    # Try recent tails so observations of a held ball before release can age out
    # quickly. At least six observations and measurable curvature are still needed.
    candidates = []
    failure = ("Waiting: drop needs usable ball-size cues and downward motion" if dropping
               else "Waiting: depth uncertain; show more of the curved flight")
    counts = [len(observations), min(18, len(observations)),
              min(12, len(observations)), min(8, len(observations))]
    if dropping:
        counts.append(min(5, len(observations)))
    for count in sorted(set(counts)):
        tail = observations[-count:]
        if np.ptp(tail[:, 0]) < minimum_span:
            continue
        if dropping:
            from tennis_tracker.drop_fit import fit_vertical_drop
            fit = fit_vertical_drop(floor.camera, floor.rotation, tail,
                                    size_observations or [], radius*2, floor.size, GRAVITY)
        else:
            fit = _fit(floor.camera, floor.rotation, tail)
        if fit is None:
            continue
        state, span, points, rmse, covariance = fit
        if rmse > (4 if dropping else 6):
            failure = "Waiting: motion does not fit free flight (R after hit/bounce)"
            continue
        impact = _impact(state, span, floor, radius)
        if impact is None:
            continue
        contact, impact_time, position, velocity = impact
        observed_world = (points-floor.offset) @ floor.rotation
        if dropping and (velocity[2] >= -0.25 or observed_world[0, 2]-observed_world[-1, 2] < 0.015):
            failure = "Waiting: drop model needs a falling ball"
            continue
        if (np.any(observed_world[:, 2] < radius-0.01)
                or np.any(observed_world[:, 2] > 15)
                or np.linalg.norm(velocity) > 100):
            continue
        remaining = impact_time - age
        if not 0 < remaining <= horizon:
            failure = "Waiting: no future landing within the time horizon"
            continue

        # Propagate fitting noise into the landing X/Y. Finite differences are
        # understandable here, and cheap for six state components.
        derivative = np.zeros((2, 6))
        valid = True
        for column in range(6):
            delta = np.zeros(6)
            delta[column] = 1e-4
            plus = _impact(state+delta, span, floor, radius)
            minus = _impact(state-delta, span, floor, radius)
            if plus is None or minus is None:
                valid = False
                break
            derivative[:, column] = (plus[0][:2]-minus[0][:2]) / (2e-4)
        if not valid:
            continue
        landing_cov = derivative @ covariance @ derivative.T
        # Twice the largest directional standard deviation is a sensitivity
        # estimate. It is NOT a guaranteed 95% physical accuracy interval.
        uncertainty = float(2*np.sqrt(max(0, np.linalg.eigvalsh(landing_cov)[-1])))
        if uncertainty > max_uncertainty:
            continue
        if not (0 <= contact[0] <= floor.width_m and 0 <= contact[1] <= floor.length_m):
            failure = "Estimated landing is outside the measured floor patch"
            continue
        contact[2] = 0  # Draw the actual floor contact, not the airborne center.
        camera_contact = floor.rotation @ contact + floor.offset
        if camera_contact[2] <= 0.1:
            continue
        x, y = floor.project([contact])[0]
        width, height = floor.size
        if not (0 <= x < width and 0 <= y < height):
            failure = "Estimated landing is outside the camera view"
            continue
        projection_jacobian = np.zeros((2, 2))
        for axis in range(2):
            delta = np.zeros(3)
            delta[axis] = 1e-4
            projection_jacobian[:, axis] = (floor.project([contact+delta])[0]
                                           - floor.project([contact-delta])[0])/(2e-4)
        pixel_cov = projection_jacobian@landing_cov@projection_jacobian.T
        pixel_uncertainty = float(2*np.sqrt(max(0, np.linalg.eigvalsh(pixel_cov)[-1])))
        future_times = np.linspace(age, impact_time, 24)
        flight = (position + future_times[:, None]*velocity
                  + 0.5*future_times[:, None]**2*np.array([0, 0, -GRAVITY]))
        camera_path = flight @ floor.rotation.T + floor.offset
        if np.any(camera_path[:, 2] <= 0.1):
            continue
        path = floor.project(flight).tolist()
        landing = Landing(float(x), float(y), remaining, path,
                          provisional=span < (0.12 if dropping else 0.35), model="physics",
                          floor_point=(float(contact[0]), float(contact[1])),
                          uncertainty_m=uncertainty, observation_count=len(tail),
                          observation_span=span, motion_model=motion_model,
                          uncertainty_px=pixel_uncertainty)
        candidates.append((rmse + uncertainty, landing))
    if not candidates:
        return None, failure
    return min(candidates, key=lambda item: item[0])[1], "Gravity-based landing estimate"
