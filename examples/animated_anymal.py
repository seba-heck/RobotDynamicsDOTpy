import os,sys
import time
import numpy as np
import yourdfpy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    # Load a URDF file
    robot = yourdfpy.URDF.load(repo_root+"/model/SimpleANYmal/simpleanymal.urdf")

    print("All joints:", robot.joint_names)
    print("Actuated joints:", robot.actuated_joint_names)
    print("Number of actuated joints:", robot.num_actuated_joints)
    print("Number of configuration DOFs:", robot.num_dofs)
    
    # callback function for animation
    def animation_callback(scene):
        # Create a time-varying configuration
        t = time.time()
        q = {
            # Floating base
            "base_x_joint": np.sin(t) * 0.25,
            "base_z_joint": np.cos(t) * 0.10,
            "base_pitch_joint": np.sin(t) * 0.3,

            # Hind legs
            "hind_hip": np.sin(t) * 0.5,
            "hind_knee": 0.0,

            # Front legs
            "front_hip": 0.0,
            "front_knee": 0.0,

            # Arm
            "shoulder": np.sin(t) * 0.5,
            "elbow": 0.0,
            "wrist": 0.0,
        }
        
        # Update the robot configuration
        robot.update_cfg(q)
    
    # Show the model with animation
    robot.show(callback=animation_callback)