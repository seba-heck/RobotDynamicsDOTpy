import numpy as np
import pinocchio as pin


PLANAR_ROWS = [0, 2, 4] # Select planar components: x, z, rotation around y
    

def _get_planar_jacobian(model,data,frame_id,planar_rows=PLANAR_ROWS):
    J = pin.getFrameJacobian(
        model, data, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J[planar_rows, :]

def _get_planar_jacobian_derivative(model,data,frame_id,planar_rows=PLANAR_ROWS):
    Jdot = pin.getFrameJacobianTimeVariation(
        model, data, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return Jdot[planar_rows, :]


def get_M(model,data,q):
    # Calculate inertia matrix M(q)
    M = pin.crba(model, data, q)
    # CRBA calculates only the upper triangular part
    M = np.triu(M) + np.triu(M, 1).T
    return M

def get_b(model,data,q,v):
    # Calculate nonlinear effects b(q, v)
    b = pin.nonLinearEffects(model, data, q, v)
    return b

def get_g(model,data,q):
    g = pin.computeGeneralizedGravity(model, data, q)
    return g

def get_J(model,data,q):
    front_foot_id = model.getFrameId("front_foot")
    hind_foot_id = model.getFrameId("hind_foot")
    ee_id = model.getFrameId("end_effector")

    # Compute joint Jacobians
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    # Jacobians
    I_J_Ff = _get_planar_jacobian(model,data,front_foot_id)
    I_J_Hf = _get_planar_jacobian(model,data,hind_foot_id)
    I_J_EE = _get_planar_jacobian(model,data,ee_id)

    return I_J_Ff,I_J_Hf,I_J_EE

def get_dJ(model,data,q,dq):
    front_foot_id = model.getFrameId("front_foot")
    hind_foot_id = model.getFrameId("hind_foot")
    ee_id = model.getFrameId("end_effector")

    # Compute time variation of joint Jacobians
    pin.computeJointJacobiansTimeVariation(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    # Jacobian derivatives
    I_Jd_Ff = _get_planar_jacobian_derivative(model,data,front_foot_id)
    I_Jd_Hf = _get_planar_jacobian_derivative(model,data,hind_foot_id)

    return I_Jd_Ff,I_Jd_Hf

def get_T(model,data,q,dq):
    front_foot_id = model.getFrameId("front_foot")
    hind_foot_id = model.getFrameId("hind_foot")
    ee_id = model.getFrameId("end_effector")

    # Forward kinematics
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    # World transformations
    I_T_Ff = data.oMf[front_foot_id].homogeneous
    I_T_Hf = data.oMf[hind_foot_id].homogeneous
    I_T_EE = data.oMf[ee_id].homogeneous

    return I_T_Ff,I_T_Hf,I_T_EE


def calc_E_pot(model,data,q):
    return pin.computePotentialEnergy(model,data,q)

def calc_E_kin(model,data,q,dq):
    return pin.computeKineticEnergy(model,data,q,dq)

def calc_H(model,data,q,dq):
    return calc_E_pot(model,data,q) + calc_E_kin(model,data,q,dq)