# RobotDynamicsDOTpy
### Robot Dynamics Exercises in Python

This repository contains exercises of the course 'Robot Dynamics from Prof. Hutter, ETH Zürich, translated into Python. A disclaimer: I'm not a lecturer or TA of the course only a former student. The course material and exam might change and differ from this.

I liked the course, it was interesting and had good presenters, but I wasn't a big fan of MATLAB. When I solved the exercises I always thought I could do everything in Python. I didn't wanted to download and learn MATLAB once again only for this course. Therefore I implemented the first exercises in Python, but I lost track of it in the later and more complex exercises. Now, I had finally the time to finish and upload some exercises. If you want to avoid MATLAB feel free to use it, but let me remind you that the midterm might still use MATLAB. Plus, my implementation uses simple time-stepping during the simulation loop and does not achieve the same performance as Simulink.

## Notes
 - NEW: **Added MuJoCo Simulation**
   To improve the performance of the robotic simulation I tried MuJoCo. I added an extra package for the Python API of MuJoCo:
   ```
   uv sync --extra mujoco
   ```
   MuJoCo is a great simulation framework for robotic applications. I converted the IRB120 model from URDF-file to a xml-file for MuJoCo, because the previous URDF-file is less stable than the xml-file model. Then I implemented a similar simulation as before, but replaced the whole time-stepping and dynamics with `mj_step`. The controller/simulation runs significantly faster and achieves higher accuracy by the hybrid controller (with the non-linear external force), but it still needs a quit small time-step for a stable simulation and does not reaches full simulation in real-time.

## TODO
 - Finish other exerices
 - test out ~~MuJoCo~~, etc...
    - implement tests for MuJoCo
    - tune and finish up MuJoCo files
    - benchmark simulations
 - extend yourdfpy