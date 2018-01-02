import prf
from prf.exceptions import UnderDefinedSystem, OverDefinedSystem
import pytest
from numpy.testing import assert_allclose


def test_under_defined():
    state0 = prf.State.define(p=1.1e6, T=300, fluid='CO2')
    state1 = prf.State.define(fluid='CO2')
    state2 = prf.State.define(T=305, fluid='CO2')

    stream0 = prf.Stream('s0', state=state0, flow_m=1)
    stream1 = prf.Stream('s1', state=state1, flow_m=None)
    stream2 = prf.Stream('s2', state=state2, flow_m=None)

    mix0 = prf.Mixer('mix0')
    mix0.link(inputs=[stream0, stream1], outputs=[stream2])
    with pytest.raises(UnderDefinedSystem) as exc:
        mix0.run()
        assert 'System is over defined' in exc.excinfo


def test_over_defined():
    # over defined pressure to equalize all
    state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state1 = prf.State.define(p=1.1e6, fluid='CO2')
    state2 = prf.State.define(p=1.2e6, T=305, fluid='CO2')

    stream0 = prf.Stream('s0', state=state0, flow_m=1)
    stream1 = prf.Stream('s1', state=state1, flow_m=2)
    stream2 = prf.Stream('s2', state=state2, flow_m=None)

    with pytest.raises(OverDefinedSystem):
        mix0 = prf.Mixer('mix0')
        mix0.link(inputs=[stream0, stream1], outputs=[stream2])
        mix0.run()

    # over defined mass flow for valve
    state0 = prf.State.define(p=100000, T=305, fluid='CO2')
    state1 = prf.State.define(p=70000, fluid='CO2')

    stream0 = prf.Stream('s0', state=state0, flow_m=2)
    stream1 = prf.Stream('s1', state=state1, flow_m=3)

    with pytest.raises(OverDefinedSystem):
        valve0 = prf.Valve('valve0')
        valve0.link(inputs=[stream0], outputs=[stream1])
        valve0.run()


def test_mixer():
    state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state1 = prf.State.define(fluid='CO2')
    state2 = prf.State.define(T=305, fluid='CO2')

    stream0 = prf.Stream('s0', state=state0, flow_m=1)
    stream1 = prf.Stream('s1', state=state1, flow_m=2)
    stream2 = prf.Stream('s2', state=state2, flow_m=None)

    mix0 = prf.Mixer('mix0')
    mix0.link(inputs=[stream0, stream1], outputs=[stream2])
    mix0.run()

    assert_allclose(stream0.flow_m, 1)
    assert_allclose(stream1.flow_m, 2)
    assert_allclose(stream2.flow_m, 3)

    assert_allclose(stream0.state.hmass(), 498833.05345178104, rtol=1e-4)
    assert_allclose(stream1.state.hmass(), 505740.4737134241, rtol=1e-4)
    assert_allclose(stream2.state.hmass(), 503438.00029393873, rtol=1e-4)


def test_tee():
    units = dict(p_units='bar', speed_units='RPM')
    fluid = dict(CO2=0.79585, R134a=0.16751, Nitrogen=0.02903, Oxygen=0.007616)
    state0 = prf.State.define(fluid=fluid)
    state1 = prf.State.define(p=7.656, T=410.4, fluid=fluid, **units)
    state2 = prf.State.define(fluid=fluid)
    stream0 = prf.Stream(name='s0', state=state0, flow_m=5.593)
    stream1 = prf.Stream(name='s1', state=state1)
    stream2 = prf.Stream(name='s2', state=state2, flow_m=0.1444)

    tee0 = prf.Tee('tee0')
    tee0.link(inputs=[stream0], outputs=[stream1, stream2])
    tee0.run()


def test_valve():
    state3 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state4 = prf.State.define(p=0.5e6, fluid='CO2')

    stream3 = prf.Stream('s3', state=state3, flow_m=None)
    stream4 = prf.Stream('s4', state=state4, flow_m=None)

    valve0 = prf.Valve('valve0', 10)
    valve0.link(inputs=[stream3], outputs=[stream4])
    valve0.run()

    assert_allclose(stream3.flow_m, 21551.930, rtol=1e-5)
    assert_allclose(stream4.flow_m, 21551.930, rtol=1e-5)
    assert_allclose(stream4.state.T(), 294.4749311434537, rtol=1e-5)

    state3 = prf.State.define(p=1e6, T=300, fluid='CO2')
    state4 = prf.State.define(p=0.5e6, fluid='CO2')

    stream3 = prf.Stream('s3', state=state3, flow_m=21551.930)
    stream4 = prf.Stream('s4', state=state4, flow_m=None)

    valve0 = prf.Valve('valve0')
    valve0.link(inputs=[stream3], outputs=[stream4])
    valve0.run()

    assert_allclose(valve0.cv, 10)
    assert_allclose(stream4.flow_m, 21551.930, rtol=1e-5)
    assert_allclose(stream4.state.T(), 294.4749311434537, rtol=1e-5)





