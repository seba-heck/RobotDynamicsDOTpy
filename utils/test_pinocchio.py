import os,sys
import yourdfpy
import pinocchio

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def show_robot(path: str):
    """Load the URDF file from given (str) path and shows it."""
    # YOURDFPY: Load a URDF file
    try:
        robot = yourdfpy.URDF.load(path)
    except FileNotFoundError:
        print(f"File ({path}) does not exists.")

    # PINOCCHIO: Load the urdf model
    model = pinocchio.buildModelFromUrdf(path)
    print("model name: " + model.name)

    # Create data required by the algorithms
    data = model.createData()
    # Sample a random configuration
    q = pinocchio.randomConfiguration(model)
    print(f"q: {q.T}")
    # Perform the forward kinematics over the kinematic tree
    pinocchio.forwardKinematics(model, data, q)
    # Print out the placement of each joint of the kinematic tree
    for name, oMi in zip(model.names, data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    # Update & visualize the robot
    robot.update_cfg(q)
    robot.show()

    return robot

if __name__ == "__main__":
    if len(sys.argv) == 1:
        file = repo_root+"/model/SimpleANYmal/simpleanymal.urdf"
    else:
        file = sys.argv[1]

    print("Showing with yourdfpy: ", file)
    show_robot(file)