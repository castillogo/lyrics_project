# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:59:40 2020

@author: casti
"""

# import pytest
from lyrics_scraping_file import souploop, lyricsmodification

#artist='Oasis'
def test_Oasis():
    assert lyricsmodification(souploop('Oasis'), 'Oasis') >= 100
