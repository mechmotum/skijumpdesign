import pytest

from ..functions import make_jump
from ..classes import InvalidJumpError


def test_problematic_jump_parameters():

    # Invalid value in arsin in LandingTransitionSurface.calc_trans_acc()
    make_jump(-26.0, 0.0, 3.0, 27.0, 0.6)

    # TODO : Fix these.
    # Invalid value in sqrt
    #make_jump(-10.0, 0.0, 30.0, 23.0, 0.2)
    #make_jump(-10.0, 0.0, 30.0, 20.0, 0.1)
    #make_jump(-25.0, 0.0, 30.0, 20.0, 0.0)  # shouldn't be able to pass in 0.0

    # Divide by zero in scipy/integrate/_ivp/rk.py
    # RuntimeWarning: divide by zero encountered in double_scalars
    #make_jump(-10.0, 10.0, 30.0, 20.0, 0.2)
    #make_jump(-45.0, 0.0, 30.0, 0.0, 0.2)

    #while loop ran more than 1000 times
    # this first situation seems to be that the acceleration at max landing
    # can't be below the threshold, thus the landing transition point ends up
    # under the parent slope surface.
    #make_jump(-45.0, 0.0, 30.0, 0.0, 0.2)
    #make_jump(-45.0, 0.0, 30.0, 0.0, 0.3)
    #make_jump(-10.0, 0.0, 30.0, 20.0, 2.0)
    #make_jump(-15.0, 0.0, 30.0, 20.0, 3.0)
    #make_jump(-15.0, 0.0, 30.0, 20.0, 2.7)

    # ValueError: x and y arrays must have at least 2 entries
    #make_jump(-10.0, 0.0, 30.0, 20.0, 2.0)
    #make_jump(-10.0, 0.0, 30.0, 20.0, 1.5)
    #make_jump(-15.0, 0.0, 30.0, 20.0, 3.0)
    #make_jump(-15.0, 0.0, 30.0, 20.0, 2.7)

    #ValueError: need at least one array to concatenate
    #make_jump(-15.0, 0.0, 30.0, 20.0, 2.8)

    # Fall height too large
    with pytest.raises(InvalidJumpError):
        make_jump(-15.0, 0.0, 30.0, 15.0, 2.7)
