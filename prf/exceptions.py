"""prf.Exceptions

This module implements prf Exceptions.
"""


class PrfError(Exception):
    """Base class for prf exceptions."""


class MassError(PrfError):
    pass
