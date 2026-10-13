"""Constrained vertical-drop fit using viewing rays and approximate ball size."""
import numpy as np


def fit_vertical_drop(camera, rotation, samples, size_observations, ball_diameter,
                      frame_size, gravity=9.81):
    times = samples[:, 0]-samples[-1, 0]
    span = float(np.ptp(times))
    tau = times/span
    upward = rotation[:, 2]
    offsets = -0.5*gravity*times[:, None]**2*upward
    rays = camera.normalized(samples[:, 1:])
    sizes = {float(t): (box, source) for t, box, source in size_observations}
    # Color silhouettes and YOLO boxes can have a systematic size difference.
    # Align their scales using nearby, nearly stationary observations before
    # fitting a drop, rather than treating the disagreement as physical depth.
    recent = [entry for entry in size_observations if samples[-1, 0]-entry[0] <= 0.6]
    yolo_sizes = [entry for entry in recent if entry[2] == "YOLO"]
    ratios = []
    for timestamp, box, source in recent:
        if source == "YOLO" or not yolo_sizes:
            continue
        match = min(yolo_sizes, key=lambda entry: abs(entry[0]-timestamp))
        if abs(match[0]-timestamp) > 0.12:
            continue
        other = match[1]
        center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
        other_center = np.array([(other[0]+other[2])/2, (other[1]+other[3])/2])
        diameter = min(box[2]-box[0], box[3]-box[1])
        other_diameter = min(other[2]-other[0], other[3]-other[1])
        if (min(diameter, other_diameter) < 8
                or np.linalg.norm(center-other_center) > max(3, other_diameter*0.1)
                or max(box[2]-box[0], box[3]-box[1])/diameter > 1.35):
            continue
        ratio = other_diameter/diameter
        if 0.65 < ratio < 1.4:
            ratios.append(ratio)
    correction = float(np.median(ratios)) if len(ratios) >= 3 else None
    spread = float(np.std(ratios)/correction) if correction else None
    if spread is not None and spread > 0.1:
        correction = None

    def adjusted_size(diameter, source):
        if source == "YOLO":
            return diameter, 0.15
        if correction is not None:
            return diameter*correction, float(np.sqrt(0.15**2+spread**2+0.05**2))
        return diameter, 0.25
    anchor_indices, depths, deviations = [], [], []
    width, height = frame_size
    focal = camera.matrix[0, 0]
    for index, sample in enumerate(samples):
        if float(sample[0]) not in sizes:
            continue
        box, source = sizes[float(sample[0])]
        x1, y1, x2, y2 = box
        diameter = min(x2-x1, y2-y1)
        if (diameter < 8 or max(x2-x1, y2-y1)/max(diameter, 1) > 2.2
                or x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1):
            continue
        diameter, fractional_error = adjusted_size(diameter, source)
        depth = focal*ball_diameter/diameter
        # Box size is a noisy cue, especially during blur/color recovery. It
        # constrains depth softly; it must not overwhelm the pixel trajectory.
        sigma = max(depth*fractional_error, focal*ball_diameter*2/diameter**2)
        anchor_indices.append(index)
        depths.append(depth)
        deviations.append(sigma)
    if len(anchor_indices) < 3:
        return None
    indices = np.array(anchor_indices)
    depths, deviations = np.array(depths), np.array(deviations)
    pixel_sigma = 1.5
    ray_sigma = pixel_sigma*np.median(depths)/focal
    # The held ball immediately before release has the same horizontal floor
    # position as a vertical drop. Use those earlier size cues as soft XY priors;
    # do not force the held interval itself to obey free-fall acceleration.
    prior_rows, prior_targets = [], []
    horizontal = rotation[:, :2].T
    for timestamp, box, source in size_observations:
        if not samples[0, 0]-0.4 <= timestamp < samples[0, 0]:
            continue
        x1, y1, x2, y2 = box
        diameter = min(x2-x1, y2-y1)
        if (diameter < 8 or max(x2-x1, y2-y1)/max(diameter, 1) > 2.2
                or x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1):
            continue
        center = ((x1+x2)/2, (y1+y2)/2)
        ray = np.r_[camera.normalized([center])[0], 1.0]
        diameter, fractional_error = adjusted_size(diameter, source)
        depth = focal*ball_diameter/diameter
        sigma = max(depth*fractional_error,
                    focal*ball_diameter*2/diameter**2)
        covariance = np.outer(ray, ray)*sigma**2
        covariance[:2, :2] += np.diag([(depth*pixel_sigma/camera.matrix[0, 0])**2,
                                      (depth*pixel_sigma/camera.matrix[1, 1])**2])
        projected_covariance = horizontal@covariance@horizontal.T+np.eye(2)*1e-9
        whitening = np.linalg.inv(np.linalg.cholesky(projected_covariance))
        prior_rows.extend(np.column_stack([whitening@horizontal, np.zeros(2)]))
        prior_targets.extend(whitening@horizontal@(ray*depth))
    prior_rows = np.asarray(prior_rows).reshape(-1, 4)
    prior_targets = np.asarray(prior_targets)
    rows, targets = [], []
    for t, ray, offset in zip(tau, rays, offsets):
        for direction in (np.array([1, 0, -ray[0]]), np.array([0, 1, -ray[1]])):
            rows.append(np.r_[direction, t*(direction@upward)]/ray_sigma)
            targets.append(-direction@offset/ray_sigma)
    for index, depth, sigma in zip(indices, depths, deviations):
        rows.append(np.array([0, 0, 1, tau[index]*upward[2]])/sigma)
        targets.append((depth-offsets[index, 2])/sigma)
    rows.extend(prior_rows)
    targets.extend(prior_targets)
    design = np.asarray(rows)
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0]/singular[-1] > 1e5:
        return None
    # Four parameters: camera-space position XYZ and vertical velocity*span.
    parameters = np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]

    def residual(value):
        points = value[:3]+tau[:, None]*upward*value[3]+offsets
        z = points[:, 2]
        if np.any(z <= 0.1):
            return None
        pixels = (points[:, :2]/z[:, None]-rays)*[camera.matrix[0, 0], camera.matrix[1, 1]]
        jacobian = np.zeros((len(points), 2, 4))
        jacobian[:, 0, 0] = camera.matrix[0, 0]/z
        jacobian[:, 1, 1] = camera.matrix[1, 1]/z
        jacobian[:, 0, 2] = -camera.matrix[0, 0]*points[:, 0]/z**2
        jacobian[:, 1, 2] = -camera.matrix[1, 1]*points[:, 1]/z**2
        jacobian[:, :, 3] = (jacobian[:, :, :3]@upward)*tau[:, None]
        anchor_j = np.column_stack([np.zeros(len(indices)), np.zeros(len(indices)),
                                   np.ones(len(indices)), tau[indices]*upward[2]])
        errors = np.r_[pixels.ravel()/pixel_sigma, (z[indices]-depths)/deviations,
                       prior_rows@value-prior_targets]
        joint_j = np.vstack([jacobian.reshape(-1, 4)/pixel_sigma, anchor_j/deviations[:, None], prior_rows])
        return errors, joint_j, points, pixels

    for _ in range(8):
        result = residual(parameters)
        if result is None:
            return None
        errors, jacobian, _, _ = result
        weights = np.sqrt(np.minimum(1, 3/np.maximum(np.abs(errors), 1e-9)))
        change = np.linalg.lstsq(jacobian*weights[:, None], -errors*weights, rcond=None)[0]
        accepted = False
        for factor in (1, 0.5, 0.25, 0.1):
            trial = residual(parameters+factor*change)
            if trial is not None and np.sum(trial[0]**2) <= np.sum(errors**2)+1e-9:
                parameters += factor*change
                accepted = True
                break
        if not accepted or np.linalg.norm(change) < 1e-7:
            break
    errors, jacobian, points, pixels = residual(parameters)
    noise_scale = max(1, float(errors@errors)/max(1, len(errors)-4))
    small_cov = np.linalg.pinv(jacobian.T@jacobian)*noise_scale
    # Convert into the same six-component state/covariance as the general fit.
    mapping = np.zeros((6, 4))
    mapping[:3, :3] = np.eye(3)
    mapping[3:, 3] = upward
    state = mapping@parameters
    covariance = mapping@small_cov@mapping.T
    rmse = float(np.sqrt(np.mean(np.sum(pixels**2, axis=1))))
    return state, span, points, rmse, covariance
