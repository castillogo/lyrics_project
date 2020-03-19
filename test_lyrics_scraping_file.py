# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:59:40 2020

@author: casti
"""

import pytest
from lyrics_scraping_file2 import souploop, lyricsmodification

def test_artist():
    ARTISTLIST = ['Aerosmith']
    for artist in ARTISTLIST:
        assert lyricsmodification(souploop(artist), artist, ARTISTLIST)[0] >= 100
