import numpy as np

from se3 import SE3


class PlanarArm:
    """Represent a three-link arm with revolute joints in the xy-plane.

    J1 is at the base origin; F1, F2, and F3 have origins at J2, J3, and
    the tool. Each frame's x-axis points along its link. Link lengths must
    be positive. Joint angles are in degrees, measured from the preceding
    link, and positive counterclockwise about the z-axis.

    forward_kinematics() finds joint positions and the tool pose from joint
    angles. geometric_inverse_kinematics() finds up to two angle solutions
    for a tool pose; numerical_inverse_kinematics() searches from one guess.
    """

    def __init__(self, link1_length, link2_length, link3_length):
        link_lengths = np.asarray(
            [link1_length, link2_length, link3_length], dtype=float
        )
        if not np.all(np.isfinite(link_lengths)) or np.any(link_lengths <= 0):
            raise ValueError("link lengths must be finite and positive")

        self.link1_length, self.link2_length, self.link3_length = link_lengths.tolist()

    @staticmethod
    def _rotation_about_z(angle_radians):
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ])
        return rotation_matrix

    def forward_kinematics(self, q1_degrees, q2_degrees, q3_degrees):
        """Return J1-to-tool positions and the tool pose in the base frame."""
        q1_radians = np.deg2rad(q1_degrees)
        q2_radians = np.deg2rad(q2_degrees)
        q3_radians = np.deg2rad(q3_degrees)

        rotation_base_from_f1 = self._rotation_about_z(q1_radians)
        rotation_f1_from_f2 = self._rotation_about_z(q2_radians)
        rotation_f2_from_f3 = self._rotation_about_z(q3_radians)

        joint2_position_in_base = np.array([
            self.link1_length * np.cos(q1_radians),
            self.link1_length * np.sin(q1_radians),
            0,
        ])
        joint3_position_in_f1 = np.array([
            self.link2_length * np.cos(q2_radians),
            self.link2_length * np.sin(q2_radians),
            0,
        ])
        tool_position_in_f2 = np.array([
            self.link3_length * np.cos(q3_radians),
            self.link3_length * np.sin(q3_radians),
            0,
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

    def geometric_inverse_kinematics(self, tool_position, tool_direction_degrees):
        """Return zero, one, or two joint-angle triples in degrees.

        Angles are in [-180, 180). When J3 is at the base and links 1 and 2
        have equal lengths, return one representative of infinitely many solutions.
        """
        tool_position = np.asarray(tool_position, dtype=float)
        if tool_position.shape != (2,) or not np.all(np.isfinite(tool_position)):
            raise ValueError("tool_position must contain two finite coordinates")
        if not np.isfinite(tool_direction_degrees):
            raise ValueError("tool_direction_degrees must be finite")

        tool_direction_radians = np.deg2rad(tool_direction_degrees)

        joint3_position = np.array([
            tool_position[0] - self.link3_length * np.cos(tool_direction_radians),
            tool_position[1] - self.link3_length * np.sin(tool_direction_radians),
        ])

        base_to_joint3_distance = np.sqrt(
            joint3_position[0]**2 + joint3_position[1]**2
        )
        distance_tolerance = 1e-10 * max(
            1.0, self.link1_length, self.link2_length, base_to_joint3_distance
        )
        minimum_reach = abs(self.link1_length - self.link2_length)
        maximum_reach = self.link1_length + self.link2_length
        if (base_to_joint3_distance < minimum_reach - distance_tolerance
                or base_to_joint3_distance > maximum_reach + distance_tolerance):
            return []

        if base_to_joint3_distance <= distance_tolerance:
            if abs(self.link1_length - self.link2_length) > distance_tolerance:
                return []
            folded_angles = np.rad2deg([
                0.0, np.pi, tool_direction_radians - np.pi
            ])
            folded_angles = (folded_angles + 180) % 360 - 180
            return [tuple(folded_angles.tolist())]

        along_line_distance = (
            self.link1_length**2 + base_to_joint3_distance**2 - self.link2_length**2
        ) / (2 * base_to_joint3_distance)

        sideways_distance_squared = self.link1_length**2 - along_line_distance**2
        squared_distance_tolerance = distance_tolerance * max(
            1.0, self.link1_length, base_to_joint3_distance
        )
        if sideways_distance_squared < -squared_distance_tolerance:
            return []
        sideways_distance = np.sqrt(max(0.0, sideways_distance_squared))

        toward_joint3_unit = joint3_position / base_to_joint3_distance

        left_perpendicular_unit = np.array([
            -toward_joint3_unit[1], toward_joint3_unit[0]
        ])

        projection_point = along_line_distance * toward_joint3_unit

        joint2_candidates = [
            projection_point + sideways_distance * left_perpendicular_unit
        ]
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

    def jacobian(self, q1_degrees, q2_degrees, q3_degrees):
        """Return changes in (x, y, tool angle) per radian of joint motion."""
        q1, q2, q3 = np.deg2rad([q1_degrees, q2_degrees, q3_degrees])
        link2_direction = q1 + q2
        tool_direction = link2_direction + q3

        link1_x_effect = -self.link1_length * np.sin(q1)
        link1_y_effect = self.link1_length * np.cos(q1)
        link2_x_effect = -self.link2_length * np.sin(link2_direction)
        link2_y_effect = self.link2_length * np.cos(link2_direction)
        link3_x_effect = -self.link3_length * np.sin(tool_direction)
        link3_y_effect = self.link3_length * np.cos(tool_direction)

        return np.array([
            [link1_x_effect + link2_x_effect + link3_x_effect,
             link2_x_effect + link3_x_effect, link3_x_effect],
            [link1_y_effect + link2_y_effect + link3_y_effect,
             link2_y_effect + link3_y_effect, link3_y_effect],
            [1.0, 1.0, 1.0],
        ])

    def numerical_inverse_kinematics(
        self, tool_position, tool_direction_degrees, initial_angles_degrees,
        *, angle_history=None,
    ):
        """Use damped Jacobian steps to find one angle triple in degrees.

        Return None if the search does not converge from the initial angles.
        Optionally append each evaluated angle triple in degrees to angle_history.
        """
        target_position = np.asarray(tool_position, dtype=float)
        initial_angles = np.asarray(initial_angles_degrees, dtype=float)
        if target_position.shape != (2,) or not np.all(np.isfinite(target_position)):
            raise ValueError("tool_position must contain two finite coordinates")
        if not np.isfinite(tool_direction_degrees):
            raise ValueError("tool_direction_degrees must be finite")
        if initial_angles.shape != (3,) or not np.all(np.isfinite(initial_angles)):
            raise ValueError("initial_angles_degrees must contain three finite angles")

        target_direction = np.deg2rad(tool_direction_degrees)
        angles = np.deg2rad(initial_angles)
        damping = 0.05
        maximum_step = 0.5
        maximum_iterations = 200
        position_tolerance = 1e-9
        direction_tolerance = 1e-9

        for iteration in range(maximum_iterations + 1):
            positions, _ = self.forward_kinematics(*np.rad2deg(angles))
            position_error = target_position - positions[-1][:2]
            direction_error = (
                target_direction - sum(angles) + np.pi
            ) % (2 * np.pi) - np.pi
            solution = tuple(((np.rad2deg(angles) + 180) % 360 - 180).tolist())
            if angle_history is not None:
                angle_history.append(solution)
            if (np.linalg.norm(position_error) < position_tolerance
                    and abs(direction_error) < direction_tolerance):
                return solution
            if iteration == maximum_iterations:
                break

            error = np.array([*position_error, direction_error])
            jacobian = self.jacobian(*np.rad2deg(angles))
            joint_step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
            step_size = np.linalg.norm(joint_step)
            if step_size > maximum_step:
                joint_step *= maximum_step / step_size
            angles += joint_step

        return None
