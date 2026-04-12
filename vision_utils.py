import cv2
import numpy as np

from runtime_config import COLOR_VERIFICATION_ENABLED, MIN_COLOR_RATIO, SQUARE_ASPECT_TOLERANCE


def undistort_point(x, y, camera_matrix, distortion):
    pts = np.array([[[x, y]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, distortion, P=camera_matrix)
    return undistorted[0, 0, 0], undistorted[0, 0, 1]


def ratio_of_target_color_and_square_shape(frame, bbox, label, write_debug_log):
    if not COLOR_VERIFICATION_ENABLED:
        return True

    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    box_width = float(x2 - x1)
    box_height = float(y2 - y1)
    aspect = box_width / max(1.0, box_height)
    if not (1.0 - SQUARE_ASPECT_TOLERANCE <= aspect <= 1.0 + SQUARE_ASPECT_TOLERANCE):
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask, debug_lines = _build_mask_and_debug(hsv, label)
    contours = []
    solidity = compactness = contour_area = 0.0
    approx = None
    angles = None

    if label == "blue_square":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 50:
            return False
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        if solidity < 0.7 or solidity > 1.0:
            return False
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
        if compactness < 0.6 or compactness > 0.95:
            return False
        if len(contours) > 3:
            return False
    elif label == "red_square":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 100:
            return False
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        if solidity < 0.85 or solidity > 1.0:
            return False
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
        if compactness < 0.75 or compactness > 0.95:
            return False
        if len(contours) != 1:
            return False
        if box_width > 0 and box_height > 0:
            aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
            if aspect_ratio > 1.3:
                return False
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) != 4:
            return False
        points = approx.reshape(4, 2)
        angles = []
        for index in range(4):
            pt1 = points[index]
            pt2 = points[(index + 1) % 4]
            pt3 = points[(index + 2) % 4]
            v1 = pt1 - pt2
            v2 = pt3 - pt2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
        if not all(80 <= angle <= 100 for angle in angles):
            return False

    color_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    ratio = color_pixels / max(1, total_pixels)

    debug_parts = [
        f"\n[{label.upper()}] bbox=({x1},{y1},{x2},{y2}) size={box_width:.0f}x{box_height:.0f}",
        *debug_lines,
        f"  Color pixels: {color_pixels}/{total_pixels} = {ratio:.3f}",
        f"  Required ratio: {MIN_COLOR_RATIO.get(label, 0.25):.3f}",
    ]
    if contours:
        debug_parts.append(f"  Contours: {len(contours)}")
        debug_parts.append(f"  Solidity: {solidity:.3f}")
        debug_parts.append(f"  Compactness: {compactness:.3f}")
        debug_parts.append(f"  Contour area: {contour_area:.1f}")
    if approx is not None:
        debug_parts.append(f"  Corners: {len(approx)}")
    if angles is not None:
        debug_parts.append(f"  Angles: {[f'{angle:.1f}' for angle in angles]}")
    write_debug_log("\n".join(debug_parts))

    return ratio >= MIN_COLOR_RATIO.get(label, 0.25)


def _build_mask_and_debug(hsv, label):
    if label == "blue_square":
        hsv_mean = np.mean(hsv, axis=(0, 1))
        base_hue_min, base_hue_max = 85, 140
        base_sat_min, base_val_min = 50, 30
        brightness = hsv_mean[2]
        if brightness < 60:
            val_min = max(15, base_val_min - 15)
            sat_min = max(30, base_sat_min - 20)
        elif brightness > 180:
            val_min = min(80, base_val_min + 30)
            sat_min = min(70, base_sat_min + 20)
        else:
            val_min = base_val_min
            sat_min = base_sat_min
        hue_center = hsv_mean[0]
        if hue_center < 100:
            hue_min = max(75, base_hue_min - 10)
            hue_max = min(150, base_hue_max + 10)
        elif hue_center > 120:
            hue_min = max(80, base_hue_min - 5)
            hue_max = min(155, base_hue_max + 15)
        else:
            hue_min, hue_max = base_hue_min, base_hue_max
        lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
        upper = np.array([hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        debug_lines = [
            f"  HSV mean: H={hsv_mean[0]:.1f}, S={hsv_mean[1]:.1f}, V={hsv_mean[2]:.1f}",
            f"  Threshold: [{hue_min},{sat_min},{val_min}] - [{hue_max},255,255]",
        ]
        return mask, debug_lines

    lower1 = np.array([0, 80, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 80, 50], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask, ["  Threshold: red dual-range HSV"]
