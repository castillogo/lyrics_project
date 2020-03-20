# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:59:40 2020

@author: casti
"""

import pytest
from lyrics_prediction_file import print_evaluations

Y_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_PRED = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def test_print_evaluations():
    assert print_evaluations(Y_TEST, Y_PRED, 'Randomforest') > 0
