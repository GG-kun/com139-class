from enum import Enum


class CharacterType(Enum):
    """An enumeration of the types of characters"""
    UNDEFINED = 0, 'UNDEFINED type. (Not in use)'
    SOLDIER = 1, 'Soldier'
    CHICKEN = 2, 'Chicken'
    T_REX = 3, 'T-Rex'
    SPARTAN = 4, 'Spartan'
    JEDI_KNIGHT = 5, 'Jedi Knight'

    def __str__(self):
        return str(self.name)
