# --------------------------------------------------
# gradient leakage attacks 梯度泄露攻击
# --------------------------------------------------

from .base_attack import BaseAttack
from .inverting_gradients import IG
from .see_through_gradients import STG
from .cocktail_party_attack import CPA
from .fgla import FGLA
from .ica_ga import ICAGA

gla_attacks = {
    "IG": IG,
    "STG": STG,
    "FGLA": FGLA,
    "ICA-GA": ICAGA,
    "CPA": CPA,
}
