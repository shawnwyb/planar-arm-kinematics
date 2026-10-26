# Planar Arm Kinematics

A three-link planar arm with SE(3)-based forward kinematics and geometric
inverse kinematics.

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
solutions = arm.inverse_kinematics((3, 1), 0)
```

Angles are in degrees. Forward kinematics returns positions from the base
through the tool and the tool's SE(3) pose. Inverse kinematics returns up to
two angle solutions for a reachable target pose.
