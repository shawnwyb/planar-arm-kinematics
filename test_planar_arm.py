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
        solutions = self.arm.geometric_inverse_kinematics((3, 1), 0)
        self.assertEqual(len(solutions), 2)
        for angles in solutions:
            self.assert_reaches(self.arm, angles, (3, 1), 0)

    def test_unreachable_and_straight_arm_targets(self):
        self.assertEqual(self.arm.geometric_inverse_kinematics((5, 0), 0), [])
        self.assertEqual(self.arm.geometric_inverse_kinematics((1, 0), 0), [])
        solutions = self.arm.geometric_inverse_kinematics((4, 0), 0)
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
                solutions = self.arm.geometric_inverse_kinematics(
                    target_position, target_direction
                )
                self.assertTrue(solutions)
                for angles in solutions:
                    self.assert_reaches(
                        self.arm, angles, target_position, target_direction
                    )

    def test_centered_wrist_returns_one_representative(self):
        centered_arm = PlanarArm(1, 1, 1)
        solutions = centered_arm.geometric_inverse_kinematics((1, 0), 0)
        self.assertEqual(solutions, [(0, -180, -180)])
        self.assert_reaches(centered_arm, solutions[0], (1, 0), 0)

    def test_nondefault_link_lengths(self):
        arm = PlanarArm(3, 2, 1)
        positions, tool_pose = arm.forward_kinematics(25, -60, 35)
        target_position = positions[-1][:2]
        target_direction = np.rad2deg(
            np.arctan2(tool_pose.T[1, 0], tool_pose.T[0, 0])
        )
        solutions = arm.geometric_inverse_kinematics(target_position, target_direction)
        self.assertEqual(len(solutions), 2)
        for angles in solutions:
            self.assert_reaches(arm, angles, target_position, target_direction)

    def test_jacobian_matches_small_forward_kinematics_change(self):
        angles_degrees = np.array([30, -70, 20])
        angle_change_radians = np.array([1e-6, -2e-6, 1.5e-6])
        before_positions, _ = self.arm.forward_kinematics(*angles_degrees)
        after_positions, _ = self.arm.forward_kinematics(
            *(angles_degrees + np.rad2deg(angle_change_radians))
        )
        actual_change = np.array([
            *(after_positions[-1][:2] - before_positions[-1][:2]),
            sum(angle_change_radians),
        ])
        predicted_change = self.arm.jacobian(*angles_degrees) @ angle_change_radians
        np.testing.assert_allclose(actual_change, predicted_change, atol=1e-10)

    def test_numerical_ik_reaches_both_solutions_from_different_guesses(self):
        solutions = [
            self.arm.numerical_inverse_kinematics((3, 1), 0, seed)
            for seed in [(30, -60, 30), (0, 60, -60)]
        ]
        self.assertTrue(all(solution is not None for solution in solutions))
        self.assertLess(solutions[0][1], 0)
        self.assertGreater(solutions[1][1], 0)
        for angles in solutions:
            positions, tool_pose = self.arm.forward_kinematics(*angles)
            np.testing.assert_allclose(positions[-1][:2], (3, 1), atol=1e-8)
            np.testing.assert_allclose(tool_pose.T[:2, 0], (1, 0), atol=1e-8)

    def test_numerical_ik_wraps_tool_direction_error(self):
        positions, _ = self.arm.forward_kinematics(0, 60, 121)
        target_position = positions[-1][:2]
        solution = self.arm.numerical_inverse_kinematics(
            target_position, -179, (0, 60, 119)
        )
        self.assertIsNotNone(solution)
        reached_positions, reached_pose = self.arm.forward_kinematics(*solution)
        np.testing.assert_allclose(
            reached_positions[-1][:2], target_position, atol=1e-8
        )
        direction_radians = np.deg2rad(-179)
        np.testing.assert_allclose(
            reached_pose.T[:2, 0],
            (np.cos(direction_radians), np.sin(direction_radians)),
            atol=1e-8,
        )

    def test_numerical_ik_returns_none_if_target_is_unreachable(self):
        self.assertIsNone(
            self.arm.numerical_inverse_kinematics((5, 0), 0, (30, -60, 30))
        )

    def test_history_records_initial_guess_and_final_solution(self):
        history = []
        solution = self.arm.numerical_inverse_kinematics(
            (3, 1), 0, (30, -60, 30), angle_history=history
        )
        self.assertIsNotNone(solution)
        self.assertGreater(len(history), 1)
        self.assertEqual(history[0], (30, -60, 30))
        self.assertEqual(history[-1], solution)


if __name__ == "__main__":
    unittest.main()
