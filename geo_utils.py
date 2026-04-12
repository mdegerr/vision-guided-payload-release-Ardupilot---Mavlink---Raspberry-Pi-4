import math


def calculate_release_distance(altitude, velocity):
    if altitude is None or altitude <= 0:
        return None
    return velocity * math.sqrt(2 * altitude / 9.81)


def distance_2d(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1 = float(lat1), float(lon1)
        lat2, lon2 = float(lat2), float(lon2)
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            return None
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return None
        if abs(lat1 - lat2) < 1e-7 and abs(lon1 - lon2) < 1e-7:
            return 0.0

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = lat2_rad - lat1_rad
        dlon = math.radians(lon2 - lon1)
        avg_lat = (lat1_rad + lat2_rad) / 2.0
        x = dlon * math.cos(avg_lat)
        y = dlat
        distance = 6378137.0 * math.sqrt(x * x + y * y)
        return round(distance, 3 if distance < 0.01 else 1)
    except Exception:
        return None


def compute_target_gps(
    cx,
    cy,
    center_x,
    center_y,
    current_lat,
    current_lon,
    current_alt,
    camera_h_fov_deg,
    camera_width,
    camera_height,
):
    fov_x_rad = math.radians(camera_h_fov_deg)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (camera_height / camera_width))
    angle_per_pixel_x = fov_x_rad / camera_width
    angle_per_pixel_y = fov_y_rad / camera_height
    angle_x = (cx - center_x) * angle_per_pixel_x
    angle_y = (cy - center_y) * angle_per_pixel_y
    target_dx = current_alt * math.tan(angle_x)
    target_dy = current_alt * math.tan(angle_y)
    delta_lat = target_dy / 111_000
    delta_lon = target_dx / (111_000 * math.cos(math.radians(current_lat)))
    return current_lat + delta_lat, current_lon + delta_lon


def compute_target_gps_with_attitude(
    cx,
    cy,
    center_x,
    center_y,
    current_lat,
    current_lon,
    current_alt,
    roll,
    pitch,
    camera_h_fov_deg,
    camera_width,
    camera_height,
):
    fov_x_rad = math.radians(camera_h_fov_deg)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (camera_height / camera_width))
    angle_per_pixel_x = fov_x_rad / camera_width
    angle_per_pixel_y = fov_y_rad / camera_height
    angle_x = (cx - center_x) * angle_per_pixel_x
    angle_y = (cy - center_y) * angle_per_pixel_y
    total_angle_x = angle_x - roll
    total_angle_y = angle_y - pitch
    target_dx = current_alt * math.tan(total_angle_x)
    target_dy = current_alt * math.tan(total_angle_y)
    delta_lat = target_dy / 111_000
    delta_lon = target_dx / (111_000 * math.cos(math.radians(current_lat)))
    return current_lat + delta_lat, current_lon + delta_lon


def compute_target_gps_with_attitude_and_heading(
    cx,
    cy,
    center_x,
    center_y,
    current_lat,
    current_lon,
    current_alt,
    roll,
    pitch,
    heading_deg,
    camera_matrix,
    camera_h_fov_deg,
    camera_width,
    camera_height,
):
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    angle_x = math.atan2(cx - center_x, fx)
    angle_y = math.atan2(cy - center_y, fy)
    heading_rad = math.radians((90 - heading_deg) % 360) if heading_deg is not None else 0.0
    effective_x = angle_x - roll
    effective_y = angle_y - pitch
    height = float(current_alt or 0.0)
    ground_distance = height * math.tan(math.sqrt(effective_x**2 + effective_y**2))
    bearing = heading_rad + math.atan2(effective_x, effective_y)
    dx = ground_distance * math.sin(bearing)
    dy = ground_distance * math.cos(bearing)
    radius = 6378137.0
    delta_lat = (dy / radius) * (180.0 / math.pi)
    delta_lon = (dx / radius) * (180.0 / math.pi) / math.cos(math.radians(current_lat))
    target_lat = current_lat + delta_lat
    target_lon = current_lon + delta_lon
    if (distance_2d(current_lat, current_lon, target_lat, target_lon) or 0.0) > 100.0:
        return compute_target_gps_with_attitude(
            cx,
            cy,
            center_x,
            center_y,
            current_lat,
            current_lon,
            current_alt,
            roll,
            pitch,
            camera_h_fov_deg,
            camera_width,
            camera_height,
        )
    return target_lat, target_lon
