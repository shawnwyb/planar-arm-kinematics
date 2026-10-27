from planar_arm import PlanarArm


if __name__ == "__main__":
    arm = PlanarArm(2, 1, 1)
    target_position = (3, 1)
    target_direction_degrees = 0

    for angles in arm.geometric_inverse_kinematics(
        target_position, target_direction_degrees
    ):
        positions_in_base, tool_pose_in_base = arm.forward_kinematics(*angles)
        print("Joint angles:", angles)
        print("Positions:", positions_in_base)
        print("Tool pose:\n", tool_pose_in_base.T)

    numerical_angles = arm.numerical_inverse_kinematics(
        target_position, target_direction_degrees, (30, -60, 30)
    )
    print("Numerical joint angles:", numerical_angles)
