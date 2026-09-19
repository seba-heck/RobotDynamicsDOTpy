import os,sys
import yourdfpy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def show_robot(path: str):
    """Load the URDF file from given (str) path and shows it."""
    # Load a URDF file
    try:
        robot = yourdfpy.URDF.load(path)
    except FileNotFoundError:
        print(f"File ({path}) does not exists.")
    # Visualize the robot
    robot.show()

    return robot

if __name__ == "__main__":
    if len(sys.argv) == 1:
        file = repo_root+"/model/IRB120/irb120.urdf"
    else:
        file = sys.argv[2]

    print("Showing with yourdfpy: ", file)
    show_robot(file)