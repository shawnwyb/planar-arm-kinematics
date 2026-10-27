# Planar Arm Kinematics

A three-link planar arm with SE(3)-based forward kinematics and geometric
and numerical inverse kinematics. An interactive Python visualizer lets you
change joint angles, explore both geometric solutions, and play back the
numerical solver's actual angle guesses.

<img width="640" height="400" alt="overview" src="https://github.com/user-attachments/assets/94234a48-a062-4a19-b711-f74305cf3aa3" />

## Setup

```bash
# Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Run the kinematics example
python main.py

# Run the tests
python -m unittest -v
```

To leave the virtual environment:

```bash
deactivate
```

The arm is defined in `planar_arm.py`. For example:

```python
from planar_arm import PlanarArm

arm = PlanarArm(2, 1, 1)
positions, tool_pose = arm.forward_kinematics(0, 90, -90)
solutions = arm.geometric_inverse_kinematics((3, 1), 0)
numerical_solution = arm.numerical_inverse_kinematics((3, 1), 0, (30, -60, 30))
```

Angles are in degrees. Forward kinematics returns positions from the base
through the tool and the tool's SE(3) pose. Geometric inverse kinematics returns
up to two angle solutions. Numerical inverse kinematics starts from an angle
guess and returns one solution, or `None` if it does not converge.
