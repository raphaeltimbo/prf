import prf
from prf.exceptions import (MassError)
import pytest
from numpy.testing import assert_allclose


def test_exceptions():
    state0 = prf.State.define(p=1.1e6, T=300, fluid='CO2')
    state1 = prf.State.define(p=1.1e6, fluid='CO2')
    state2 = prf.State.define(p=1.1e6, T=305, fluid='CO2')

    stream0 = prf.Stream(state=state0, flow_m=1)
    stream1 = prf.Stream(state=state1, flow_m=None)
    stream2 = prf.Stream(state=state2, flow_m=None)

    mix0 = prf.Mixer()
    mix0.link(inputs=[stream0, stream1], outputs=[stream2])
    with pytest.raises(MassError) as exc:
        mix0.run()
        assert 'More than one' in exc.excinfo


def test_mixer():
    state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state1 = prf.State.define(p=1.1e6, fluid='CO2')
    state2 = prf.State.define(p=1.2e6, T=305, fluid='CO2')

    stream0 = prf.Stream(state=state0, flow_m=1)
    stream1 = prf.Stream(state=state1, flow_m=2)
    stream2 = prf.Stream(state=state2, flow_m=None)

    mix0 = prf.Mixer()
    mix0.link(inputs=[stream0, stream1], outputs=[stream2])
    mix0.run()

    assert_allclose(stream0.flow_m, 1)
    assert_allclose(stream1.flow_m, 2)
    assert_allclose(stream2.flow_m, 3)

    assert_allclose(stream0.state.hmass(), 498833.05345178104, rtol=1e-4)
    assert_allclose(stream1.state.hmass(), 502853.62131474464, rtol=1e-4)
    assert_allclose(stream2.state.hmass(), 501513.4320281485, rtol=1e-4)




