"""
This is a program for making a song prediction
according to the lyrcis available in
file output.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")

tv = TfidfVectorizer()
m = MultinomialNB()
ros = RandomOverSampler()

lyrics_df = pd.read_csv('output.csv')
lyrics_df['singer_number'] = lyrics_df['singer_number'].astype(int)
lyricswords = lyrics_df['lyrics_words'].to_list()

def print_evaluations(y_train, y_pred, model):

    """
    This function summaries all scores and
    makes a confusion matrix heatman
    """
    print(f'How does model {model} score:')
    print(f"The accuracy of the model is: {round(accuracy_score(y_train, y_pred), 3)}")
    print(f"The precision of the model is: {round(precision_score(y_train, y_pred, average='weighted'), 3)}")
    print(f"The recall of the model is: {round(recall_score(y_train, y_pred, average='weighted'), 3)}")
    print(f"The f1-score of the model is: {round(f1_score(y_train, y_pred, average='weighted'), 3)}")

    #print confusion matrix
    plt.figure(figsize=(15, 15))
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax);
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix %s' % (model));
    ax.xaxis.set_ticklabels([i for i in
                             range(len(ynumbers))]); ax.yaxis.set_ticklabels([i
                                  for i in range(len(ynumbers))])
    return accuracy_score(y_train, y_pred, model)

if __name__ == '__main__':

    print('')
    print("""
This is a program for making a song prediction
according to the lyrcis available in
file output.csv
    """)
    print('')
    print("""
Please save your lyrics as a txt file
in this folder and write the name
of the file here:
    """)
    print('')

    song = input()
    song = str(song)
    with open ("%s.txt" % (song), "r") as myfile:
        data=myfile.readlines()
    data=str(data)
    data= re.sub(r"\\n", ' ', data)

    print('')
    print("""
These are your lyrics:
    """)
    print('')
    print(data)

    tv.fit(lyricswords)
    tv_vectors = tv.transform(lyricswords)
    y= lyrics_df['singer'].to_list()
    ynumbers = lyrics_df['singer_number'].to_list()
    m.fit(tv_vectors,ynumbers)
    new_song=[data]
    tv_vec = tv.transform(new_song)

    #simple naive bayes

    print('')
    print("""
This is a simple naive bayes predcition
without input optimization:
    """)
    print('')
    print("""
(please check dictionary printed at the end of the
run to see which artist corresponds
to which artistnumber)
    """)
    print('')
    print("""
Your song belongs most probably
to this artistnumber:
    """)
    print('')

    print(m.predict(tv_vec))

    print('')
    print("""
These are the probabilities that your
song belongs to each artistnumber:
    """)
    print('')

    print(m.predict_proba(tv_vec))

    print('')

    df = pd.DataFrame(zip(lyricswords, y), columns=['lyricswords', 'ynumbers'])
    tv_vectors.shape

    y = df['ynumbers']
    X = df[['lyricswords']]

    X_resample, y_resample = ros.fit_resample(X, y)
    X_resample

    cv = CountVectorizer(ngram_range=(1,1))
    cv.fit(lyricswords)

    word_vectors = cv.transform(lyricswords)
    cv.get_feature_names()
    df2 = pd.DataFrame(word_vectors.todense(), columns=cv.get_feature_names())
    X = df2
    y = df['ynumbers']

    print('')
    print("""
These are the train-test predicitions
for a baseline model:
    """)
    print('')

    split=0.1
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        ynumbers, 
                                                        random_state=10, 
                                                        test_size=split)

    #Baseline model

    ypred_bl = [0] * X_train.shape[0]
    print_evaluations(y_train, ypred_bl, 'Baseline')
    new_df = pd.concat([X, y], axis = 1)
    new_df.groupby('ynumbers').size()
    new_df.groupby('ynumbers').size()[1]/new_df.shape[0]*100
    X = new_df.iloc[:,:-1]
    y = new_df.ynumbers

    # simple Random forest model
    print('')
    print("""
These are the results of the
random forest evaluation:
    """)

    rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=10)
    rf.fit(X_train, y_train)
    ypred_rf = rf.predict(X_test)
    rf2=rf

    print('')
    print("""
This is the random forest prediction
for the artist number for your song:
    """)
    print('')

    print(rf.predict(tv_vec))

    print("""
These are the probabilities that
your song belongs to each artistnumber:
    """)
    print('')

    print(rf.predict_proba(tv_vec))

    print('')
    print("""
These are the random forest evaluations
for the train-test split:
    """)
    print('')

    print_evaluations(y_test, ypred_rf, 'RandomForest')

    # Random oversampling model

    ros = RandomOverSampler(random_state=10)
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    np.unique(y_ros, return_counts=True)
    rf2.fit(X_ros, y_ros)
    ypred_ros = rf2.predict(X_test)

    print('')
    print("""
This is the random oversampling prediction
of the artist number with random forest
evaluation for your song:
    """)
    print('')

    print(rf2.predict(tv_vec))

    print('')
    print("""
These are the probabilities that
your song belongs to each artistnumber:
    """)
    print('')
    print(rf2.predict_proba(tv_vec))
    print('')
    print("""
These are the random oversampling
evaluations with the train-test split:
    """)
    print('')
    print_evaluations(y_test, ypred_ros, 'RandomOversampling')


    y= lyrics_df['singer'].to_list()
    ynumbers = lyrics_df['singer_number'].to_list()
    artistlistfinal = dict(zip(y, ynumbers))

    print('')
    print("""
This is the code for the artists
and the belonging artistnumbers:
    """)
    print(artistlistfinal)
    print('')
    print("""
These are the heatmaps for the confusion
matrix of each different evaluation:
    """)
