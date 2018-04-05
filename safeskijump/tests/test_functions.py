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
    #make_jump(-25.0, 0.0, 30.0, 20.0, 0.0)

    # Fall height too large
    with pytest.raises(InvalidJumpError):
        make_jump(-15.0, 0.0, 30.0, 15.0, 2.7)
