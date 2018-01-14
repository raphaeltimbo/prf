import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from .point import *
from .state import *
from warnings import warn
from copy import copy


__all__ = ['Impeller', 'NonDimPoint']


class Impeller:
    """
    Impeller instance is initialized with the dimensional curve (init_curve).
    The created instance will hold the dimensional curve used in instantiation
    a non dimensional curve generated from the given dimensional curve and
    another dimensional curve based on current suction condition and speed.
    The impeller also has a current point that depends on the flow.
    Current condition can be set after instantiation.

    Parameters
    ----------
    init_curves : list
        List with curves instances used to instantiate the impeller.
    b : float
        Impeller width (m).
    D : float
        Impeller diameter (m).
    e : float
        Impeller roughness.
        Defaults to 0.87 um.

    Returns
    -------
    non_dim_points : list
        List with non dimensional point instances.

    Attributes
    ----------

    Examples
    --------
    """
    def __init__(self, init_curves, b, D, e=0.87e-6,
                 suc=None, speed=None, flow_v=None):

        class Config:
            """config class for impeller"""
            def __init__(self):
                # TODO create config
                pass

        if isinstance(init_curves, list) and isinstance(init_curves[0], Curve):
            self.init_curves = init_curves
        elif isinstance(init_curves, Curve):
            self.init_curves = [init_curves]
        elif isinstance(init_curves, list) and isinstance(init_curves[0], Point):
            self.init_curves = [Curve(init_curves)]
        elif isinstance(init_curves, Point):
            self.init_curves = [Curve(init_curves)]
        else:
            raise TypeError('Must be a point, curve or list of points - curves')

        # set each curve as an attribute
        for c in self.init_curves:
            setattr(self, 'init_curve_' + f'{c.speed:.0f}', c)

        self.config = Config()

        self.points = list(self.init_curves[0])
        self.b = b
        self.D = D
        self.e = e
        self.non_dim_points = []
        for point in self.points:
            self.non_dim_points.append(NonDimPoint.from_impeller(self, point))

        # impeller current state
        self._suc = self.points[0].suc if suc is None else suc
        self._speed = self.points[0].speed if speed is None else speed
        self._flow_v = self.points[0].flow_v if flow_v is None else flow_v

        # the current points and curve
        self.new_points = None
        self.new_curve = None
        self.not_valid_points = None
        self.suc_p_curve = None
        self.suc_T_curve = None
        self.disch_p_curve = None
        self.disch_T_curve = None
        self.head_curve = None
        self.eff_curve = None
        self.power_curve = None
        self.disch = None
        self._current_point = None
        self._calc_new()

    def __repr__(self):
        return (
            'Impeller:'
            + ' Diameter {} m,'.format(self.D)
            + ' Width {} m'.format(self.b)
        )

    def __str__(self):
        points = ''
        for p in self.points:
            points += '\n {!r}'.format(p)
        return (
            'Impeller:'
            + ' Diameter {} m,'.format(self.D)
            + ' Width {} m'.format(self.b)
            + points
        )

    @property
    def suc(self):
        return self._suc

    @suc.setter
    def suc(self, new_suc):
        self._suc = new_suc
        self._calc_new()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, new_speed):
        self._speed = new_speed
        self._calc_new()

    @property
    def flow_v(self):
        return self._flow_v

    @flow_v.setter
    def flow_v(self, new_flow_v):
        self._flow_v = new_flow_v
        self._calc_new()

    @property
    def current_point(self):
        return self._current_point

    @current_point.setter
    def current_point(self, new_point):
        self._flow_v = new_point.flow_v
        self._speed = new_point.speed
        self._suc = new_point.suc
        # self._current_point is set by _calc_new()
        # This is done so that cases for flow_v, speed
        # and suc are also handled. Setting it here
        # could cause an infinite recursion
        self._calc_new()

    def _calc_new(self):
        """Calculate new points.
        
        This function is called when the object is instantiated.
        It will also be called when properties such as suction state,
        flow or speed are changed.
        """
        self.new_points = [self.new_point(self.suc, self.speed, i)
                           for i in range(len(self.points))]

        self.new_curve = Curve(self.new_points)
        self.suc_p_curve = self.new_curve.suc_p_curve
        self.suc_T_curve = self.new_curve.suc_T_curve
        self.disch_p_curve = self.new_curve.disch_p_curve
        self.disch_T_curve = self.new_curve.disch_T_curve
        self.head_curve = self.new_curve.head_curve
        self.eff_curve = self.new_curve.eff_curve
        self.power_curve = self.new_curve.power_curve

        current_disch_p = self.disch_p_curve(self.flow_v)
        current_disch_T = self.disch_T_curve(self.flow_v)
        current_disch = copy(self.suc)
        current_disch.update(CP.PT_INPUTS, current_disch_p, current_disch_T)

        self.disch = current_disch
        self._current_point = Point(suc=self.suc,
                                    disch=current_disch,
                                    flow_v=self.flow_v,
                                    speed=self.speed)
        self.check_similarity()

    def check_similarity(self):
        """Verify similarity.
        
        This function will verify the similarity between points stored and 
        new points that are generated based on the non dimensional points.
        
        """
        not_valid_points = {}

        for i, p in enumerate(self.new_points):
            if not all([p.mach_comparison['valid'],
                        p.reynolds_comparison['valid'],
                        p.volume_ratio_comparison['valid']]):
                not_valid_points['p' + str(i)] = {'Mach': p.mach_comparison,
                                                  'Reynolds': p.reynolds_comparison,
                                                  'Volume ration': p.volume_ratio_comparison}

        if len(not_valid_points) > 0:
            pts = ', '.join(not_valid_points)
            warn('Following points out of similarity: %s' % pts)

        self.not_valid_points = not_valid_points

    def flow_coeff(self, flow_m=None, suc=None, speed=None, point=None):
        """Flow coefficient.

        Calculates the flow coefficient for a point given the mass flow,
        suction state and speed.

        Parameters 
        ---------- 
        point
        flow_m : float
            Mass flow (kg/s)
        suc : prf.State
            Suction state.
        speed : float
            Speed in rad/s.
        point : int 
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and speed.
            
        Returns 
        ------- 
        flow_coeff : float
            Flow coefficient (non dimensional).

        Examples 
        -------- 

        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            flow_m = point.flow_m
            suc = point.suc
            speed = point.speed

        v = 1 / suc.rhomass()
        u = self.tip_speed(speed)

        # 3.2.5 ISO-5389
        flow_coeff = (flow_m * v * 4 /
                      (np.pi * self.D**2 * u))

        return flow_coeff

    def tip_speed(self, speed=None, point=None):
        """Impeller tip speed.

        Calculates the impeller tip speed for a given speed.

        Parameters
        ----------
        speed : float
            Speed in rad/s.
        point : int 
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and speed.
 
        Returns
        -------
        tip_speed : float
            Impeller tip speed in m/s.

        Examples
        --------

        """
        # TODO check dimensions and units
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            speed = point.speed

        u = speed * self.D / 2

        return u

    def head_coeff(self, head=None, speed=None, point=None):
        """Head coefficient.

        Calculates the head coefficient given a head and speed.

        Parameters
        ----------
        point
        head : float
            Head in J/kg.
        speed : float
            Speed in rad/s.
        point : int 
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and speed.
 
        Returns
        -------
        head_coeff : float
            Head coefficient (non dimensional).
        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]

            head = point.head
            speed = point.speed

        u = self.tip_speed(speed)

        head_coeff = 2 * head / u**2  # 3.2.6 ISO-5389

        return head_coeff

    def mach(self, suc=None, speed=None, point=None):
        """Mach number.
         
        Parameters
        ----------
        suc : prf.State
            Suction state.
        speed : float
            Speed in rad/s.
        point : int 
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and speed.
        
        Returns
        -------
        mach : float
            Mach number.
        """
        if point is None:
            point = self.current_point
            suc = point.suc
            speed = point.speed

        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            speed = point.speed

        u = self.tip_speed(speed)
        a = suc.speed_sound()

        mach = u / a  # 3.2.3 ISO-5389

        return mach

    def reynolds(self, suc=None, speed=None, point=None):
        """Reynolds number.
         
        Parameters
        ----------
        suc : prf.State
            Suction state.
        speed : float
            Speed in rad/s.
        point : int 
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and speed.
        
        Returns
        -------
        Reynolds : float
            Reynolds number.
        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            speed = point.speed

        u = self.tip_speed(speed)
        b = self.b
        v = suc.viscosity() / suc.rhomass()  # E.115 ISO-5389

        reynolds = u * b / v  # 3.2.4 ISO-5389

        return reynolds

    def volume_ratio(self, suc=None, disch=None, point=None):
        """Volume ratio.
         
        Parameters
        ----------
        suc : prf.State
            Suction state.
        disch : prf.State
            Discharge state.
        point : prf.Point, int
            Point or index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and disch.
        
        Returns
        -------
        Volume ratio : float
            Volume ratio.
        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            disch = point.disch

        volume_ratio = suc.rhomass() / disch.rhomass()

        return volume_ratio

    def compare_dimensionless(self, dimensionless, point=None, other_point=None):
        """Compare dimensionless number.

        This method compares dimensionless number between points and return
        a pandas series with limits and values from the comparison.

        Parameters
        ----------
        other : prf.Point
            Other point.

        dimensionless : str
            Dimensionless number to be compared.
            Options are: volume_ratio, mach and reynolds

        point : int
            Index for a point inside the impeller instance.
            If the point is provided, no need to provide suc and disch.

        Returns
        -------
        S : pd.Series
            pandas series with limits and values from the comparison.
        """
        args = {'volume_ratio', 'mach', 'reynolds'}

        if dimensionless not in args:
            raise ValueError(f'Argument not valid: {dimensionless}. '
                             f'Should be in {args}.')

        if dimensionless == 'volume_ratio':
            ratio = (self.volume_ratio(point=point)
                     / self.volume_ratio(point=other_point))

            lower_limit = 0.95
            upper_limit = 1.05

            if lower_limit < ratio < upper_limit:
                valid = True
            else:
                valid = False

            d = {'ratio': ratio, 'valid': valid, 'lower_limit': lower_limit,
                 'upper_limit': upper_limit}

            return pd.Series(d)

        elif dimensionless == 'mach':
            mach_sp = self.mach(point=point)
            mach_t = self.mach(point=other_point)

            if mach_sp < 0.214:
                lower_limit = -mach_sp
                upper_limit = -0.25 * mach_sp + 0.286
            elif 0.215 < mach_sp < 0.86:
                lower_limit = 0.266 * mach_sp - 0.271
                upper_limit = -0.25 * mach_sp + 0.286
            else:
                lower_limit = -0.042
                upper_limit = 0.07

            diff = mach_sp - mach_t

            if lower_limit < diff < upper_limit:
                valid = True
            else:
                valid = False

            d = {'diff': diff, 'valid': valid, 'lower_limit': lower_limit,
                 'upper_limit': upper_limit}

            return pd.Series(d)

        elif dimensionless == 'reynolds':
            reynolds_sp = self.reynolds(point=point)
            reynolds_t = self.reynolds(point=other_point)

            x = (reynolds_sp/1e7)**0.3

            if 9e4 < reynolds_sp < 1e7:
                upper_limit = 100**x
            elif 1e7 < reynolds_sp:
                upper_limit = 100
            else:
                upper_limit = 100

            if 9e4 < reynolds_sp < 1e6:
                lower_limit = 0.01**x
            elif 1e6 < reynolds_sp:
                lower_limit = 0.1
            else:
                lower_limit = 0.1

            ratio = reynolds_t/reynolds_sp

            if lower_limit < ratio < upper_limit:
                valid = True
            else:
                valid = False

            d = {'ratio': ratio, 'valid': valid, 'lower_limit': lower_limit,
                 'upper_limit': upper_limit}

            return pd.Series(d)

    @convert_to_base_units
    def new_point(self, suc, speed, idx, **kwargs):
        """Curve.

        Calculates a new point based on the given suction state and speed.
        """
        # TODO check the closest flow. Add new arg point?

        point_old = self.points[idx]
        non_dim_point = self.non_dim_points[idx]

        rho = suc.rhomass()
        u = self.tip_speed(speed)

        # calculate the mass flow
        phi = non_dim_point.flow_coeff
        flow_v = phi * np.pi * self.D**2 * u / 4
        flow_m = flow_v * rho

        # calculate new head and efficiency
        psi = non_dim_point.head_coeff
        head = psi * u**2 / 2
        eff = non_dim_point.eff

        point_new = Point(flow_m=flow_m, speed=speed, suc=suc, head=head, eff=eff)

        point_new.mach_comparison = self.compare_dimensionless(
            'mach', point_old, point_new)
        point_new.reynolds_comparison = self.compare_dimensionless(
            'reynolds', point_old, point_new)
        point_new.volume_ratio_comparison = self.compare_dimensionless(
            'volume_ratio', point_old, point_new)

        return point_new

    @classmethod
    def load_from_excel(cls, file, **kwargs):
        """Load curve from excel file.

        Parameters
        ----------
        file : excel file
            Excel file with the following sheets:
            'TEST-PYTHON' and 'SPECIFIED-PYTHON'

        Returns
        -------
        imp : prf.Impeller
            Impeller object with the test curve and current
            condition according to specified.
        """

        def comp_from_df(df):
            # get fluids from dataframe
            fluids = []
            for col in df.columns:
                if col in fluid_list:
                    fluids.append(col)
            # get composition for each point
            test_points_comp = {}

            for p in df.T:
                point_comp = {}
                for f in fluids:
                    point_comp[f] = df[f][p]
                test_points_comp[p] = point_comp

            return test_points_comp

        # create point from df
        def point_from_df(df, **kwargs):
            comp = comp_from_df(df)

            for p in df.T:
                # point used to calibrate k values from the seal
                if p == 'PONTO10':
                    continue
                if not df['ps'][p] == 0:
                    # create suction state
                    ps = df['ps'][p]
                    Ts = df['Ts'][p]
                    suc = State.define(p=ps, T=Ts, fluid=comp[p], **kwargs)
                    # create suction state
                    pd = df['pd'][p]
                    Td = df['Td'][p]
                    disch = State.define(p=pd, T=Td, fluid=comp[p], **kwargs)

                    flow = df['mass_flow'][p]
                    speed = df['speed'][p]

                    yield Point(speed=speed, flow_m=flow, suc=suc, disch=disch, **kwargs)

        sec = kwargs.get('sec', '-SEC1')
        test_points_data = pd.read_excel(file,
                                         sheetname=('TEST-PYTHON' + sec))
        spec_points_data = pd.read_excel(file,
                                         sheetname=('SPECIFIED-PYTHON' + sec))

        points_test = []
        for point in point_from_df(test_points_data, **kwargs):
            points_test.append(point)

        curve_test = Curve(points_test)

        D = spec_points_data['D'][0]
        b = spec_points_data['b'][0]

        # TODO change impeller state to spec conditions
        point_sp = [p for p in point_from_df(spec_points_data, **kwargs)]
        point_sp = point_sp[0]

        imp = cls(curve_test, b, D)
        imp.current_point = point_sp

        # TODO point for seal calibration

        return imp


class NonDimPoint:
    def __init__(self, *args, **kwargs):
        # calculate non dimensional curve
        self.flow_coeff = kwargs.get('flow_coeff')
        self.head_coeff = kwargs.get('head_coeff')
        self.eff = kwargs.get('eff')

    def __repr__(self):
        return (
            '\nNon Dimensional Point: '
            + '\n Flow Coefficient : {:10.5}'.format(self.flow_coeff)
            + '\n Head Coefficient : {:10.5}'.format(self.head_coeff)
            + '\n Efficiency       : {:10.5}'.format(self.eff)
        )

    @classmethod
    def from_impeller(cls, impeller, point):
        # flow coefficient
        # calculate non dim curve and append
        flow_coeff = impeller.flow_coeff(point.flow_m,
                                         point.suc,
                                         point.speed)
        head_coeff = impeller.head_coeff(point.head,
                                         point.speed)
        eff = point.eff

        return cls(flow_coeff=flow_coeff, head_coeff=head_coeff, eff=eff)

