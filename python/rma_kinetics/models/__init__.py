"""
Kinetic models for the released markers of activity (RMAs)
for constitutive or drug-induced reporter expression.

Contains the following models:

- Constitutive - a constitutively expressed synthetic serum reporter
- TetOff - serum reporter expressed under the tetracycline responsive operator
- Chemogenetic - neuronal activity induced + doxycycline gated serum reporter expression
- Dox - doxycycline pharmacokinetic model
- CNO - clozapine-N-oxide/clozapine pharmacokinetic model
"""

__all__ = ["constitutive", "tetoff", "dox", "cno"]

