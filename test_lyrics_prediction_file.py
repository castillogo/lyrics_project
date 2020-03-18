# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:59:40 2020

@author: casti
"""

# import pytest
from lyrics_prediction_file import print_evaluations

#artist='Oasis'
def test_print_evaluations():
    assert print_evaluations(y_test, ypred_rf, 'Randomforest') >= 0.3
