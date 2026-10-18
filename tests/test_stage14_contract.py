"""Synthetic-only numeric invariants for fixed stage14."""
import numpy as np
from tools import stage14_change as C


def test_change_features_ignore_static_embedding_offset():
    rng=np.random.default_rng(5);z=rng.normal(size=(20,8));t=np.arange(20)+.5
    np.testing.assert_allclose(C.change_features({'z':z,'t':t}),C.change_features({'z':z+7.,'t':t}),atol=1e-12)


def test_change_gradient_direction_and_finite_difference():
    rng=np.random.default_rng(7);bags=[rng.normal(size=(10,4)) for _ in range(4)];y=[1,1,0,0]
    b=np.arange(5)*.01;f=lambda x:C.objective(x,bags,y)
    _,g=f(b)
    numeric=[]
    for j in range(5):
        d=np.zeros(5);d[j]=1e-6;numeric.append((f(b+d)[0]-f(b-d)[0])/2e-6)
    np.testing.assert_allclose(g,numeric,atol=1e-7)
