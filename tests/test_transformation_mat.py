import numpy as np
import pytest
from RobotArmIRB120 import jointToTransform01, jointToTransform12, jointToTransform23, jointToTransform34, jointToTransform45, jointToTransform56, getTransformI0, getTransform6E


N_TESTS = 1000


def _testing_jointToTransform_shape(func, q):
    T_ = func(q)
    return T_.shape == (4,4)

def _testing_jointToTransform_oz(func, q):
    T_ = func(q)
    return T_[3,3] == 1.0 and ( T_[3,0:3] == np.zeros(3) ).all()

@pytest.mark.parametrize("func", [jointToTransform01, jointToTransform12, jointToTransform23, jointToTransform34, jointToTransform45, jointToTransform56])
def test_jointToTransform_form(func):
    print(f"Testing {func.__name__} ...", end=" ")
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        assert _testing_jointToTransform_shape(func,q)
        assert _testing_jointToTransform_oz(func, q)
    print("passed.")

@pytest.mark.parametrize("func", [getTransformI0, getTransform6E])
def test_getTransform_form(func):
    print(f"Testing {func.__name__} ...", end=" ")
    T_ = func()

    assert T_.shape == (4,4), "Error: Transformation matrix has wrong shape!"
    assert T_[3,3] == 1.0, "Error: Transformation matrix missing 1.0!"
    assert ( T_[3,0:3] == np.zeros(3) ).all(), "Error: Transformation matrix missing zeros!"
    print("passed.")

for func in [jointToTransform01, jointToTransform12, jointToTransform23, jointToTransform34, jointToTransform45, jointToTransform56]:
    test_jointToTransform_form(func)


for func in [getTransformI0, getTransform6E]:
    test_getTransform_form(func)