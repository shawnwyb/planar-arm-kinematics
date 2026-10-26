import numpy as np
from se3 import SE3

def rotation_about_z(angle_radians):
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1],
    ])
    return rotation_matrix

def forward_kinematics(q1_degrees, q2_degrees, q3_degrees):
    q1_radians = np.deg2rad(q1_degrees)
    q2_radians = np.deg2rad(q2_degrees)
    q3_radians = np.deg2rad(q3_degrees)

    rotation_base_from_f1 = rotation_about_z(q1_radians)
    rotation_f1_from_f2 = rotation_about_z(q2_radians)
    rotation_f2_from_f3 = rotation_about_z(q3_radians)

    joint2_position_in_base = np.array([
        2 * np.cos(q1_radians), 2 * np.sin(q1_radians), 0
    ])
    joint3_position_in_f1 = np.array([
        np.cos(q2_radians), np.sin(q2_radians), 0
    ])
    tool_position_in_f2 = np.array([
        np.cos(q3_radians), np.sin(q3_radians), 0
    ])

    transform_base_from_f1 = SE3(rotation_base_from_f1, joint2_position_in_base)
    transform_f1_from_f2 = SE3(rotation_f1_from_f2, joint3_position_in_f1)
    transform_f2_from_f3 = SE3(rotation_f2_from_f3, tool_position_in_f2)

    transform_base_from_f2 = transform_base_from_f1.compose(transform_f1_from_f2)
    transform_base_from_f3 = transform_base_from_f2.compose(transform_f2_from_f3)

    positions_in_base = [
        np.array([0, 0, 0]),
        transform_base_from_f1.T[:3, 3],
        transform_base_from_f2.T[:3, 3],
        transform_base_from_f3.T[:3, 3],
    ]
    return positions_in_base, transform_base_from_f3

def inverse_kinematics(tool_position, tool_direction_degrees,
                       link1_length, link2_length, link3_length):
    """Return zero, one, or two joint-angle triples in degrees.

    Angles are in [-180, 180). When J3 is at the base and links 1 and 2
    have equal lengths, return one representative of infinitely many solutions.
    """
    tool_position = np.asarray(tool_position, dtype=float)
    link_lengths = np.asarray(
        [link1_length, link2_length, link3_length], dtype=float
    )
    if tool_position.shape != (2,) or not np.all(np.isfinite(tool_position)):
        raise ValueError("tool_position must contain two finite coordinates")
    if not np.isfinite(tool_direction_degrees):
        raise ValueError("tool_direction_degrees must be finite")
    if not np.all(np.isfinite(link_lengths)) or np.any(link_lengths <= 0):
        raise ValueError("link lengths must be finite and positive")

    tool_direction_radians = np.deg2rad(tool_direction_degrees)

    joint3_position = np.array([
        tool_position[0] - link3_length * np.cos(tool_direction_radians),
        tool_position[1] - link3_length * np.sin(tool_direction_radians),
    ])

    base_to_joint3_distance = np.sqrt(
        joint3_position[0]**2 + joint3_position[1]**2
    )
    distance_tolerance = 1e-10 * max(
        1.0, link1_length, link2_length, base_to_joint3_distance
    )
    if (base_to_joint3_distance > link1_length + link2_length + distance_tolerance
            or base_to_joint3_distance < abs(link1_length - link2_length) - distance_tolerance):
        return []

    if base_to_joint3_distance <= distance_tolerance:
        if abs(link1_length - link2_length) > distance_tolerance:
            return []
        folded_angles = np.rad2deg([
            0.0, np.pi, tool_direction_radians - np.pi
        ])
        folded_angles = (folded_angles + 180) % 360 - 180
        return [tuple(folded_angles.tolist())]

    along_line_distance = (
        link1_length**2 + base_to_joint3_distance**2 - link2_length**2
    ) / (2 * base_to_joint3_distance)

    sideways_distance_squared = link1_length**2 - along_line_distance**2
    squared_distance_tolerance = distance_tolerance * max(
        1.0, link1_length, base_to_joint3_distance
    )
    if sideways_distance_squared < -squared_distance_tolerance:
        return []
    sideways_distance = np.sqrt(max(0.0, sideways_distance_squared))

    toward_joint3_unit = joint3_position / base_to_joint3_distance

    left_perpendicular_unit = np.array([
        -toward_joint3_unit[1], toward_joint3_unit[0]
    ])

    projection_point = along_line_distance * toward_joint3_unit

    joint2_candidates = [projection_point + sideways_distance * left_perpendicular_unit]
    if sideways_distance > distance_tolerance:
        joint2_candidates.append(
            projection_point - sideways_distance * left_perpendicular_unit
        )

    solutions = []
    for joint2_position in joint2_candidates:
        link2_direction_in_base = np.arctan2(
            joint3_position[1] - joint2_position[1],
            joint3_position[0] - joint2_position[0],
        )
        joint1_angle = np.arctan2(joint2_position[1], joint2_position[0])
        joint2_angle = link2_direction_in_base - joint1_angle
        joint3_angle = tool_direction_radians - link2_direction_in_base
        angles_degrees = np.rad2deg([
            joint1_angle, joint2_angle, joint3_angle
        ])
        angles_degrees = (angles_degrees + 180) % 360 - 180
        solutions.append(tuple(angles_degrees.tolist()))

    return solutions

if __name__ == "__main__":
    positions, tool_pose = forward_kinematics(90, -90, 0)
    print("Positions:", positions)
    print("Tool pose:\n", tool_pose.T)
