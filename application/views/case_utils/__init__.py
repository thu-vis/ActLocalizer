from .case_base import CaseBase
from .case_thumos import CaseTHUMOS

from ..utils.config_utils import config

def get_case_util(dataname, case_mode):
    if dataname.lower() == config.thumos19.lower():
        return CaseTHUMOS(case_mode)
    else:
        raise ValueError("unsupport dataname {}".format(dataname))