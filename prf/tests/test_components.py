import prf
from prf.exceptions import MassError, OverDefinedWarning
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


def test_warnings():
    # over defined pressure to equalize all
    state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state1 = prf.State.define(p=1.1e6, fluid='CO2')
    state2 = prf.State.define(p=1.2e6, T=305, fluid='CO2')

    stream0 = prf.Stream(state=state0, flow_m=1)
    stream1 = prf.Stream(state=state1, flow_m=2)
    stream2 = prf.Stream(state=state2, flow_m=None)

    with pytest.warns(OverDefinedWarning):
        mix0 = prf.Mixer()
        mix0.link(inputs=[stream0, stream1], outputs=[stream2])
        mix0.run()

    # over defined pressure to set outlet to lowest inlet
    state0 = prf.State.define(p=1.e6, T=300, fluid='CO2')
    state1 = prf.State.define(p=1.0e6, fluid='CO2')
    state2 = prf.State.define(1.0e6, T=305, fluid='CO2')

    stream0 = prf.Stream(state=state0, flow_m=1)
    stream1 = prf.Stream(state=state1, flow_m=2)
    stream2 = prf.Stream(state=state2, flow_m=None)

    mix0 = prf.Mixer()
    mix0.pressure_assignment.set_to(1)
    mix0.link(inputs=[stream0, stream1], outputs=[stream2])
    mix0.setup()
    mix0.run()


def test_mixer():
    state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state1 = prf.State.define(fluid='CO2')
    state2 = prf.State.define(T=305, fluid='CO2')

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
    assert_allclose(stream1.state.hmass(), 505740.4737134241, rtol=1e-4)
    assert_allclose(stream2.state.hmass(), 503438.00029393873, rtol=1e-4)


def test_valve():
    state3 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state4 = prf.State.define(p=0.5e6, fluid='CO2')

    stream3 = prf.Stream(state=state3, flow_m=None)
    stream4 = prf.Stream(state=state4, flow_m=None)

    valve0 = prf.Valve(10)
    valve0.link(inputs=[stream3], outputs=[stream4])
    valve0.run()

    assert_allclose(stream3.flow_m, 107637.87833389692, rtol=1e-5)
    assert_allclose(stream4.flow_m, 107637.87833389692, rtol=1e-5)
    assert_allclose(stream4.state.T(), 294.4749311434537, rtol=1e-5)

    state3 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state4 = prf.State.define(p=0.5e6, fluid='CO2')

    stream3 = prf.Stream(state=state3, flow_m=107637.87833389692)
    stream4 = prf.Stream(state=state4, flow_m=None)

    valve0 = prf.Valve(10)
    valve0.link(inputs=[stream3], outputs=[stream4])
    valve0.run()

    assert_allclose(stream4.flow_m, 107637.87833389692, rtol=1e-5)
    assert_allclose(valve0.total_mass, 107637.87833389692, rtol=1e-5)
    assert_allclose(stream4.state.T(), 294.4749311434537, rtol=1e-5)





