from .baseball import IconBaseballPitch
from .basketball import IconBasketballDunk
from .billiards import IconBilliards
from .weightlifting import IconWeightlifting
from .cricket_bowling import IconCricketBowling
from .cricket_shot import IconCricketShot
from .diving import IconDiving
from .frisbee import IconFrisbeeCatch
from .golf import IconGolfSwing
from .hammer import IconHammerThrow
from .highjump import IconHighJump
from .javelin import IconJavelinThrow
from .longjump import IconLongJump
from .pole_vault import IconPoleVault
from .shotput import IconShotput
from .soccer import IconSoccerPenalty
from .tennis import IconTennisSwing
from .throwdiscus import IconThrowDiscus
from .volleyball import IconVolleyballSpiking
from .icon_base import save_square

# ['BaseballPitch', 
#  'BasketballDunk',
#  'Billiards',     
#  'CleanAndJerk',  
#  'CricketBowling',
#  'CricketShot',
#  'Diving',
#  'FrisbeeCatch',
#  'GolfSwing',
#  'HammerThrow',
#  'HighJump',
#  'JavelinThrow',
#  'LongJump',
#  'PoleVault',
#  'Shotput',
#  'SoccerPenalty',
#  'TennisSwing',
#  'ThrowDiscus',
#  'VolleyballSpiking']

def get_icon_class(name):
    if name == "BaseballPitch":
        return IconBaseballPitch()
    elif name == "BasketballDunk": # too small
        return IconBasketballDunk()
    elif name == "Billiards": # add the cue
        return IconBilliards()
    elif name == "CleanAndJerk":
        return IconWeightlifting()
    elif name == "CricketBowling": # to separate
        return IconCricketBowling()
    elif name == "CricketShot": # to separate
        return IconCricketShot()
    elif name == "Diving": # add the water
        return IconDiving()
    elif name == "FrisbeeCatch": # tracking the same man
        return IconFrisbeeCatch()
    elif name == "GolfSwing": 
        return IconGolfSwing()
    elif name == "HammerThrow":
        return IconHammerThrow()
    elif name == "HighJump":
        return IconHighJump()
    elif name == "JavelinThrow":
        return IconJavelinThrow()
    elif name == "LongJump":
        return IconLongJump()
    elif name == "PoleVault":
        return IconPoleVault()
    elif name == "Shotput":
        return IconShotput()
    elif name == "SoccerPenalty":
        return IconSoccerPenalty()
    elif name == "TennisSwing":
        return IconTennisSwing()
    elif name == "ThrowDiscus":
        return IconThrowDiscus()
    elif name == "VolleyballSpiking":
        return IconVolleyballSpiking()
    else:
        raise ValueError("unsupport class")
