from enum import Enum


class Activity(Enum):
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3
    A = 4
    B = 5
    C = 6
    D = 7
    E = 8
    F = 9


PAD = Activity.PAD.value
UNK = Activity.UNK.value
SOS = Activity.SOS.value
EOS = Activity.EOS.value
A = Activity.A.value
B = Activity.B.value
C = Activity.C.value
D = Activity.D.value
E = Activity.E.value
F = Activity.F.value
