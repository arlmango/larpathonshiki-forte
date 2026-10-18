"""Synthetic all40 split contract; no real data is loaded."""
import numpy as np
from tools.stage13_all40 import stratified_folds


def test_all40_stratified_group_partition():
    y=[1]*21+[0]*19
    folds=stratified_folds(y)
    assert sorted(i for f in folds for i in f)==list(range(40))
    assert len(folds)==5 and folds==stratified_folds(y)
    for held in folds:
        train=set(range(40))-set(held)
        assert not train.intersection(held)
        assert 4<=sum(y[i] for i in held)<=5
        assert 3<=sum(1-y[i] for i in held)<=4
        assert {y[i] for i in train}=={0,1}
