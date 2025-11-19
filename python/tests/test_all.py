import pytest
from rma_kinetics import models, solvers

T0 = 0
T1 = 168
DT = 1

solver = solvers.Dopri5()

def test_model_creation():
    constitutive_model = models.constitutive.Model(0.2, 0.6, 0.007)
