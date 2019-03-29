import numpy as np
from numpy.testing import assert_allclose

from ..utils import vel2speed


def test_vel2speed():

    io = [((0.0, 0.0), (0.0, 0.0)),
          ((1.0, 0.0), (1.0, 0.0)),
          ((1.0, 1.0), (np.sqrt(2.0), np.pi / 4)),
          ((0.0, 1.0), (1.0, np.pi / 2)),
          ((-1.0, 1.0), (np.sqrt(2.0), 3 * np.pi / 4)),
          ((-1.0, 0.0), (1.0, np.pi)),
          ((-1.0, -1.0), (np.sqrt(2.0), -3 * np.pi / 4)),
          ((0.0, -1.0), (1.0, -np.pi / 2)),
          ((1.0, -1.0), (np.sqrt(2.0), -np.pi / 4))]

    for ins, outs in io:

        speed, angle = vel2speed(*ins)
        assert_allclose(speed, outs[0])
        assert_allclose(angle, outs[1])
