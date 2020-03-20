"""
This is a program for making a song prediction
according to the lyrcis available in
file output.csv
"""
import re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings("ignore")

TV = TfidfVectorizer()
M = MultinomialNB()
ROS = RandomOverSampler()

LYRICS_DF = pd.read_csv('output.csv')
LYRICS_DF['singer_number'] = LYRICS_DF['singer_number'].astype(int)
LYRICSWORDS = LYRICS_DF['lyrics_words'].to_list()

def print_evaluations(Y_TRAIN, Y_PRED, model):

    """
    This function summaries all scores and
    makes a confusion matrix heatman
    """
    print(f'How does model {model} score:')
    print(f"The accuracy of the model is: {round(accuracy_score(Y_TRAIN, Y_PRED), 3)}")
    print(f"The precision of the model is: {round(precision_score(Y_TRAIN, Y_PRED, average='weighted'), 3)}")
    print(f"The recall of the model is: {round(recall_score(Y_TRAIN, Y_PRED, average='weighted'), 3)}")
    print(f"The f1-score of the model is: {round(f1_score(Y_TRAIN, Y_PRED, average='weighted'), 3)}")

    #print confusion matrix
    plt.figure(figsize=(15, 15))
    Cm = confusion_matrix(Y_TRAIN, Y_PRED)
    print(Cm)
    Ax = plt.subplot()
    sns.heatmap(Cm, annot=True, ax=Ax)
    Ax.set_xlabel('Predicted labels')
    Ax.set_ylabel('True labels')
    Ax.set_title('Confusion Matrix %s' % (model))
    return accuracy_score(Y_TRAIN, Y_PRED, model)

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

    SONG = input()
    SONG = str(SONG)
    with open("%s.txt" % (SONG), "r") as myfile:
        DATA = myfile.readlines()
    DATA = str(DATA)
    DATA = re.sub(r"\\n", ' ', DATA)

    print('')
    print("""
These are your lyrics:
    """)
    print('')
    print(DATA)

    TV.fit(LYRICSWORDS)
    TV_VECTORS = TV.transform(LYRICSWORDS)
    Y = LYRICS_DF['singer'].to_list()
    YNUMBERS = LYRICS_DF['singer_number'].to_list()
    M.fit(TV_VECTORS, YNUMBERS)
    NEW_SONG = [DATA]
    TV_VEC = TV.transform(NEW_SONG)

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

    print(M.predict(TV_VEC))

    print('')
    print("""
These are the probabilities that your
song belongs to each artistnumber:
    """)
    print('')

    print(M.predict_proba(TV_VEC))

    print('')

    DF = pd.DataFrame(zip(LYRICSWORDS, Y), columns=['LYRICSWORDS', 'YNUMBERS'])

    Y = DF['YNUMBERS']
    X = DF[['LYRICSWORDS']]

    X_RESAMPLE, Y_RESAMPLE = ROS.fit_resample(X, Y)

    CV = CountVectorizer(ngram_range=(1, 1))
    CV.fit(LYRICSWORDS)

    WORD_VECTORS = CV.transform(LYRICSWORDS)
    CV.get_feature_names()
    DF2 = pd.DataFrame(WORD_VECTORS.todense(), columns=CV.get_feature_names())
    X = DF2
    Y = DF['YNUMBERS']

    print('')
    print("""
These are the train-test predicitions
for a baseline model:
    """)
    print('')

    SPLIT = 0.1
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X,
                                                        YNUMBERS,
                                                        random_state=10,
                                                        test_size=SPLIT)

    #Baseline model

    YPRED_BL = [0] * X_TRAIN.shape[0]
    print_evaluations(Y_TRAIN, YPRED_BL, 'Baseline')
    NEW_DF = pd.concat([X, Y], axis=1)
    NEW_DF.groupby('YNUMBERS').size()
    #NEW_DF.groupby('YNUMBERS').size()[1]/NEW_DF.shape[0]*100
    X = NEW_DF.iloc[:, :-1]
    Y = NEW_DF.YNUMBERS

    # simple Random forest model
    print('')
    print("""
These are the results of the
random forest evaluation:
    """)

    RF = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=10)
    RF.fit(X_TRAIN, Y_TRAIN)
    YPRED_RF = RF.predict(X_TEST)
    RF2 = RF

    print('')
    print("""
This is the random forest prediction
for the artist number for your song:
    """)
    print('')

    print(RF.predict(TV_VEC))

    print("""
These are the probabilities that
your song belongs to each artistnumber:
    """)
    print('')

    print(RF.predict_proba(TV_VEC))

    print('')
    print("""
These are the random forest evaluations
for the train-test split:
    """)
    print('')

    print_evaluations(Y_TEST, YPRED_RF, 'RandomForest')

    # Random oversampling model

    ROS = RandomOverSampler(random_state=10)
    X_ROS, Y_ROS = ROS.fit_resample(X_TRAIN, Y_TRAIN)
    np.unique(Y_ROS, return_counts=True)
    RF2.fit(X_ROS, Y_ROS)
    YPRED_ROS = RF2.predict(X_TEST)

    print('')
    print("""
This is the random oversampling prediction
of the artist number with random forest
evaluation for your song:
    """)
    print('')

    print(RF2.predict(TV_VEC))

    print('')
    print("""
These are the probabilities that
your song belongs to each artistnumber:
    """)
    print('')
    print(RF2.predict_proba(TV_VEC))
    print('')
    print("""
These are the random oversampling
evaluations with the train-test split:
    """)
    print('')
    print_evaluations(Y_TEST, YPRED_ROS, 'RandomOversampling')


    Y = LYRICS_DF['singer'].to_list()
    YNUMBERS = LYRICS_DF['singer_number'].to_list()
    ARTISTLISTFINAL = dict(zip(Y, YNUMBERS))

    print('')
    print("""
This is the code for the artists
and the belonging artistnumbers:
    """)
    print(ARTISTLISTFINAL)
    print('')
    print("""
These are the heatmaps for the confusion
matrix of each different evaluation:
    """)
