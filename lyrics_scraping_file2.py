"""This is a program to import lyrics from different artists"""

import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

ARTISTLIST = []
X = 1

HEADER = "singer,singer_number,lyrics_words"
print(HEADER,  file=open("output.csv", 'w'))

def souploop(artist):
    """This is the main web scraping loop"""
    LYRICSLIST = []

    for n in range(1):
        """This is the specific/secondary web scraping loop
        I decided to leave the for loop so that the range can be easily changen to 10
        and the url to url="https://www.metrolyrics.com/%s-alpage-%s.html" % (artist, n)"""

        url = "https://www.metrolyrics.com/%s-lyrics.html" % (artist)
        r = requests.get(url)
        SOUP = [a["href"] for a in BeautifulSoup(r.content,
                                                 features="lxml").find_all("a", "title", href=True)]
        VERSES = [BeautifulSoup(requests.get(url).content,
                                features="lxml").find_all("p", "verse") for url in SOUP]
        for i, verse in enumerate(VERSES):
            LYRICSLIST.append([v.text for v in verse])

    return str(LYRICSLIST)


def lyricsmodification(lyrics, artist, artistlist):
    """This is the lyrics cleaning loop"""
    ARTISTLIST = artistlist
    lyrics = re.sub(r"<..>", ' ', lyrics)
    lyrics = re.sub(r"<...>", ' ', lyrics)
    lyrics = re.sub(r"<.*>", ' ', lyrics)
    lyrics = re.sub(r"\\n", ' linebreakdummy ', lyrics)
    lyrics = "initialdummy" + lyrics
    lyrics = re.sub(r"[^a-zA-Z0-9'-]+", ' ', lyrics)
    lyrics = re.sub(r"\s+", ' ', lyrics)
    lyrics = re.sub(r"linebreakdummy", " \n artistdummy, factorizingdummy, ", lyrics)
    lyrics = re.sub(r"initialdummy", " artistdummy, factorizingdummy, ", lyrics)
    lyrics = re.sub(r"\s'", " ", lyrics)
    lyrics = re.sub(r"'\s", " ", lyrics)
    lyrics = lyrics + "finaldummy"
    lyrics = re.sub(r"finaldummy", '', lyrics)
    lyrics = re.sub(r"artistdummy", artist, lyrics)
    lyrics = re.sub(r"factorizingdummy", str(ARTISTLIST.index(artist)), lyrics)

    print(lyrics, file=open("%s.csv" % (artist), 'w'))
    print(lyrics, file=open("output.csv", 'a'))
    print('csv file has been created with Lyrics from: ' + artist)

    final_file_size = int(Path("output.csv").stat().st_size)
    print(final_file_size)

    return final_file_size, artistlist

if __name__ == '__main__':

    print('')

    print("This is a program to create a list of lyrics from different artists")
    print('')

    while True:
        ARTISTNUMBER = str(X)
        print('Give name of artist' + ARTISTNUMBER + ' or write break to stop:')
        ARTIST = input()
        ARTIST = str(ARTIST)
        ARTIST = re.sub(r'\ ', '-', ARTIST)
        X = X+1
        print('')
        if ARTIST == 'break':
            break
        ARTISTLIST.append(ARTIST)



    print('the list of artists you made is:')
    print('')
    print(ARTISTLIST)
    print('')
    print('scraping lyrics...')
    print('')

    for artist in ARTISTLIST:
        lyricsmodification(souploop(artist), artist, ARTISTLIST)

    print('')
    print("all lyrics have been merged to output.csv")
    print('')
    print("Thank you")
