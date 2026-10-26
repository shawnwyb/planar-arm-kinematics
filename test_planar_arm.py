import unittest

import numpy as np

from planar_arm import PlanarArm


class InverseKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.arm = PlanarArm(2, 1, 1)

    def assert_reaches(self, arm, angles, target_position, target_direction_degrees):
        positions, tool_pose = arm.forward_kinematics(*angles)
        direction_radians = np.deg2rad(target_direction_degrees)
        expected_x_axis = [np.cos(direction_radians), np.sin(direction_radians)]
        np.testing.assert_allclose(positions[-1][:2], target_position, atol=1e-9)
        np.testing.assert_allclose(tool_pose.T[:2, 0], expected_x_axis, atol=1e-9)

    def test_two_arm_shapes_reach_the_same_pose(self):
        solutions = self.arm.inverse_kinematics((3, 1), 0)
        self.assertEqual(len(solutions), 2)
        for angles in solutions:
            self.assert_reaches(self.arm, angles, (3, 1), 0)

    def test_unreachable_and_straight_arm_targets(self):
        self.assertEqual(self.arm.inverse_kinematics((5, 0), 0), [])
        self.assertEqual(self.arm.inverse_kinematics((1, 0), 0), [])
        solutions = self.arm.inverse_kinematics((4, 0), 0)
        self.assertEqual(len(solutions), 1)
        self.assert_reaches(self.arm, solutions[0], (4, 0), 0)

    def test_forward_then_inverse_recovers_target_pose(self):
        configurations = [(30, -70, 20), (120, 50, -100), (-135, 90, 45)]
        for original_angles in configurations:
            with self.subTest(original_angles=original_angles):
                positions, tool_pose = self.arm.forward_kinematics(*original_angles)
                target_position = positions[-1][:2]
                target_direction = np.rad2deg(
                    np.arctan2(tool_pose.T[1, 0], tool_pose.T[0, 0])
                )
                solutions = self.arm.inverse_kinematics(
                    target_position, target_direction
                )
                self.assertTrue(solutions)
                for angles in solutions:
                    self.assert_reaches(
                        self.arm, angles, target_position, target_direction
                    )

    def test_centered_wrist_returns_one_representative(self):
        centered_arm = PlanarArm(1, 1, 1)
        solutions = centered_arm.inverse_kinematics((1, 0), 0)
        self.assertEqual(solutions, [(0, -180, -180)])
        self.assert_reaches(centered_arm, solutions[0], (1, 0), 0)

    def test_nondefault_link_lengths(self):
        arm = PlanarArm(3, 2, 1)
        positions, tool_pose = arm.forward_kinematics(25, -60, 35)
        target_position = positions[-1][:2]
        target_direction = np.rad2deg(
            np.arctan2(tool_pose.T[1, 0], tool_pose.T[0, 0])
        )
        solutions = arm.inverse_kinematics(target_position, target_direction)
        self.assertEqual(len(solutions), 2)
        for angles in solutions:
            self.assert_reaches(arm, angles, target_position, target_direction)


if __name__ == "__main__":
    unittest.main()
