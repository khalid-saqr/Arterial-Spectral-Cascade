from copy import deepcopy
from .study import STUDY_CONFIG

def default_study_config():
    """Return an independent mutable copy of the default study configuration."""
    return deepcopy(STUDY_CONFIG)

__all__=["default_study_config"]
