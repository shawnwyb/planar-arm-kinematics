import numpy as np
from se3 import SE3

def build_rotation_matrix(q):
    R = np.array([[np.cos(q), -np.sin(q), 0],
                  [np.sin(q), np.cos(q), 0],
                  [0, 0, 1]])
    return R

def forward_kinematics(q1, q2, q3):
    q1 = np.deg2rad(q1)
    q2 = np.deg2rad(q2)
    q3 = np.deg2rad(q3)

    R_F1 = build_rotation_matrix(q1)
    R_F2 = build_rotation_matrix(q2)
    R_F3 = build_rotation_matrix(q3)

    t_F1 = np.array([2*np.cos(q1), 2*np.sin(q1), 0])
    t_F2 = np.array([np.cos(q2), np.sin(q2), 0])
    t_F3 = np.array([np.cos(q3), np.sin(q3), 0])

    Tbase_F1 = SE3(R_F1, t_F1)
    TF1_F2 = SE3(R_F2, t_F2)
    TF2_F3 = SE3(R_F3, t_F3)

    Tbase_F2 = Tbase_F1.compose(TF1_F2)
    Tbase_F3 = Tbase_F2.compose(TF2_F3)

    positions = [
        np.array([0, 0, 0]),
        Tbase_F1.T[:3, 3],
        Tbase_F2.T[:3, 3],
        Tbase_F3.T[:3, 3],
    ]
    return positions, Tbase_F3

if __name__ == "__main__":
    positions, tool_pose = forward_kinematics(90, -90, 0)
    print("Positions:", positions)
    print("Tool pose:\n", tool_pose.T)
