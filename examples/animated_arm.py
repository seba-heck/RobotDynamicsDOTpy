import os,sys
import time
import numpy as np
import yourdfpy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    # Load a URDF file
    robot = yourdfpy.URDF.load(repo_root+"/model/IRB120/irb120.urdf")
    
    # callback function for animation
    def animation_callback(scene):
        # Create a time-varying configuration
        t = time.time()
        q = np.array([
            np.sin(t) * 0.5,
            np.sin(t * 1.5) * 0.5,
            np.sin(t * 2.0) * 0.5,
            0,
            0,
            0,
        ])
        
        # Update the robot configuration
        robot.update_cfg(q)
    
    # Show the model with animation
    robot.show(callback=animation_callback)