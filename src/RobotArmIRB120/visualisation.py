"""
VISUALISATION FUNCTIONS  for YOURDFPY
These function load the URDF file of an robot and visualises them
via yourdfpy.

Author: Sebastian Heckers
"""
import os,sys
import time
import yourdfpy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def viz_cfg(q):
    """Load the URDF file from given (str) path and shows it."""
    path = repo_root+"/model/IRB120/irb120.urdf"
    # Load a URDF file
    try:
        robot = yourdfpy.URDF.load(path)
    except FileNotFoundError:
        print(f"File ({path}) does not exists.")
    
    # Update the robot configuration
    robot.update_cfg(q)
    
    # Visualize the robot
    robot.show()

    return robot

def viz_hist(hist, dt=None):
    # Load a URDF file
    robot = yourdfpy.URDF.load(repo_root+"/model/IRB120/irb120.urdf")

    idx = 0
    
    # callback function for animation
    def animation_callback(scene):
        nonlocal idx
        # Create a time-varying configuration
        if dt is not None: time.sleep(dt)
        q = hist[idx]
        idx = (idx + 1)%len(hist)
        
        # Update the robot configuration
        robot.update_cfg(q)
    
    # Show the model with animation
    robot.show(callback=animation_callback)