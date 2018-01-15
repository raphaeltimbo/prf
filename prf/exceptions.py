"""prf.Exceptions

This module implements prf Exceptions.
"""


class PrfError(Exception):
    """Base class for prf exceptions."""


class UnderDefinedSystem(PrfError):
    pass


class OverDefinedSystem(PrfError):
    pass


class PrfWarning(Warning):
    """Base class for warnings."""


class OverDefinedWarning(PrfWarning):
    pass

