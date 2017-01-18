class Impeller:
    def __init__(self, flow, head, efficiency):
        """
        Impeller instance is initialized with the non-dimensional curves.
        """
        self.flow = flow
        self.head = head
        self.efficiency = efficiency

    @classmethod
    def from_dimensional_curves(cls, flow, head, efficiency,
                                ps, Ts, pd,
                                fluid):
        """
        Constructor to initialize an impeller from a dimensional curve.
        """
        # TODO calculate non dimensional curves
        return cls(flow, head, efficiency)