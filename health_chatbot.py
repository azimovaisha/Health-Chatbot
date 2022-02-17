# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2021
# Project Part 4
# Aisha Azimova
# =========================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import re
import pickle as pkl
import string

# ASK PROFESSOR IF THIS IS OK
import warnings
warnings.filterwarnings("ignore")

import csv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import nltk


# Before running code that makes use of Word2Vec, we will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from our professor
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (lexicon) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    lexicon = df['Lexicon'].values.tolist()
    label = df['Label'].values.tolist()
    return lexicon, label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    if token in word2vec:
        word_vector = np.array(word2vec[token])

    return word_vector


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: Two strings (a name, and a date of birth formatted as MM/DD/YY)
#
# This function extracts a name and date of birth, if available, from an input
# string using regular expressions.  Names are assumed to be UTF-8 strings of
# 2-4 consecutive camel case tokens, and dates of birth are assumed to be
# formatted as MM/DD/YY.  If a name or a date of birth can not be found in the
# string, return an empty string ("") in its place.
def extract_user_info(user_input):
    name = ""
    dob = ""

    findingname = re.search("(^|\s)([A-Z]([a-zA-Z]|[.,−&'-])*)(\s)([A-Z]([a-zA-Z]|[.,−&'-])*(\s|$)){1,3}", user_input)

    if findingname:
        name = findingname.group().strip()

    findingDOB = re.search("(\s|^)((0[1-9])|(1[0-2]))/((0[1-9])|(1[0-9])|(2[0-9])|(3[0-1]))/([0-9][0-9])(\s|$)",
                           user_input)

    if findingDOB:
        dob = findingDOB.group().strip()

    return name, dob


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing(user_input), see project statement for more details
# user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    # Initialize modified_input to be the same as the original user input
    modified_input = user_input

    tokens = get_tokens(user_input)
    no_punctuation = []
    for t in tokens:
        if t not in string.punctuation:
            no_punctuation.append(t.lower())
    modified_input = ' '.join(no_punctuation)
    return modified_input


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)

    preprocessed = preprocessing(user_input)
    tokens = get_tokens(preprocessed)
    inputLen = len(tokens)

    for token in tokens:
        embedding = np.add(embedding, w2v(word2vec, token))

    embedding = embedding / inputLen

    return embedding


# Function: vectorize_train(training_documents)
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None

    for i in range(len(training_documents)):
        training_documents[i] = preprocessing(training_documents[i])

    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train



# Function: vectorize_test(vectorizer, user_input)
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Initialize the TfidfVectorizer model and document-term matrix
    tfidf_test = None

    processed = preprocessing(user_input)

    tfidf_test = vectorizer.transform([processed])

    return tfidf_test



# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = None
    svm = None
    mlp = None

    logistic = LogisticRegression()
    svm = LinearSVC()
    mlp = MLPClassifier()

    return logistic, svm, mlp


# Function: train_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    naive = GaussianNB()

    naive.fit(training_data.toarray(), training_labels)

    return naive


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):

    training_data = []
    for doc in training_documents:
        training_data.append(string2vec(word2vec, doc))

    model.fit(np.array(training_data), training_labels)

    return model


# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0

    tokens = nltk.tokenize.word_tokenize(user_input)

    for e in tokens:
        if e in string.punctuation:
            tokens.remove(e)

    num_words = len(tokens)

    return num_words


# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0

    sentences = nltk.tokenize.sent_tokenize(user_input)

    sumOfLengths = 0

    for s in sentences:
        sumOfLengths += count_words(s)

    wps = sumOfLengths / (len(sentences))

    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []

    tokens = nltk.tokenize.word_tokenize(user_input)
    tagged_input = nltk.pos_tag(tokens)

    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    for e in tagged_input:
        if (e[1] == 'PRP'):
            num_pronouns += 1
            num_prp += 1
        elif (e[1] == 'PRP$' or e[1] == 'WP' or e[1] == 'WP$'):
            num_pronouns += 1
        elif (e[1] == 'DT'):
            num_articles += 1
        elif (e[1] == 'VBD' or e[1] == 'VBN'):
            num_past += 1
        elif e[1] == 'MD':
            num_future += 1
        elif e[1] == 'IN':
            num_prep += 1


    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0

    negations = ['no', 'not', 'never', 'n’t', 'n\'t']

    tokens = nltk.tokenize.word_tokenize(user_input)

    for e in tokens:
        if e in string.punctuation:
            tokens.remove(e)

    for e in tokens:
        if e in negations:
            num_negations += 1

    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    correlatesDone = 0

    # On the second thought, this way was very inefficent and I might come back to fix it later
    prio = {}

    prio['num_words'] = num_words
    prio['wps'] = wps
    prio['num_pronouns'] = num_pronouns
    prio['num_prp'] = num_prp
    prio['num_articles'] = num_articles
    prio['num_past'] = num_past
    prio['num_future'] = num_future
    prio['num_prep'] = num_prep
    prio['num_negations'] = num_negations

    convertedToFloat = prio

    for e in convertedToFloat:
        convertedToFloat[e] += 0.0

    sortedFloats = dict(sorted(convertedToFloat.items(), key=lambda e: e[1], reverse=True))
    del sortedFloats['num_words']
    del sortedFloats['wps']

    # Getting the ones with given thresholds out of the way
    if (num_words > num_words_threshold):
        correlatesDone += 1
        informative_correlates.append(psychological_correlates["num_words"])
    if (wps > wps_threshold):
        correlatesDone += 1
        informative_correlates.append(psychological_correlates["wps"])

    while (correlatesDone != 3):
        currMax = max(sortedFloats.values())

        if (sortedFloats['num_pronouns'] == currMax):
            informative_correlates.append(psychological_correlates["num_pronouns"])
            correlatesDone += 1
            sortedFloats['num_pronouns'] = -1
        elif (sortedFloats['num_prp'] == currMax):
            informative_correlates.append(psychological_correlates["num_prp"])
            correlatesDone += 1
            sortedFloats['num_prp'] = -1
        elif (sortedFloats['num_articles'] == currMax):
            informative_correlates.append(psychological_correlates["num_articles"])
            correlatesDone += 1
            sortedFloats['num_articles'] = -1
        elif (sortedFloats['num_past'] == currMax):
            informative_correlates.append(psychological_correlates["num_past"])
            correlatesDone += 1
            sortedFloats['num_past'] = -1
        elif (sortedFloats['num_future'] == currMax):
            informative_correlates.append(psychological_correlates["num_future"])
            correlatesDone += 1
            sortedFloats['num_future'] = -1
        elif (sortedFloats['num_prep'] == currMax):
            informative_correlates.append(psychological_correlates["num_prep"])
            correlatesDone += 1
            sortedFloats['num_prep'] = -1
        elif (sortedFloats['num_negations'] == currMax):
            informative_correlates.append(psychological_correlates["num_negations"])
            correlatesDone += 1
            sortedFloats['num_negations'] = -1

    return informative_correlates


# ***** New in Project Part 4! *****
# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    # *** Replace the line below with your updated welcome message from Project Part 1 ***
    print("Howdy, partner. Welcome to the Wild West of my Health Chat Bot. Let's start out by getting acquainted.")

    return "get_info"


# ***** New in Project Part 4! *****
# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and date of
# birth, and then processes the user's response to extract that information.
def get_info_state():
    # Request the user's name and date of birth, and accept a user response of
    # arbitrary length
    # *** Replace the line below with your updated message from Project Part 1 ***
    user_input = input("What is your name and date of birth? Enter this information in the form: First Last MM/DD/YY\n")

    # Extract the user's name and date of birth
    name, dob = extract_user_info(user_input)
    print("Thanks {0}!  I'll make a note that you were born on {1}".format(name, dob))

    return "health_check"


# ***** New in Project Part 4! *****
# Function: health_check_state(name, dob, model)
# model: The trained classification model used for predicting health status
# word2vec: OPTIONAL; The pretrained Word2Vec model
# first_time (bool): indicates whether the state is active for the first time. HINT: use this parameter to determine next state.
# Returns: A string indicating the next state
#
# This function implements a state that asks the user to describe their health,
# and then processes their response to predict their current health status.
def health_check_state(model, word2vec, first_time):
    # Check the user's current health
    user_input = input("\nHow are you feeling today? Any symptoms bothering you?\n")

    # Predict whether the user is healthy or unhealthy

    for x in get_tokens(user_input):
        w2v_test = string2vec(word2vec, x)
        label = model.predict(w2v_test.reshape(1, -1))
        if label==1:
            break


    if label == 0:
        print("Great!  It sounds like you're healthy.")
    elif label == 1:
        print("Oh no!  It sounds like you're unhealthy.")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    if (first_time == True):
        return "stylistic_analysis"
    else:
        return "check_next_state"


# ***** New in Project Part 4! *****
# Function: stylistic_analysis_state()
# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state():
    user_input = input("\nI'd also like to do a quick stylistic analysis. What's on your mind today?\n")

    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print("Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    return "check_next_state"


# ***** New in Project Part 4! *****
# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the health check
# ("health_check"), or redo the stylistic analysis
# ("stylistic_analysis").
def check_next_state():
    next_state = ""

    # [YOUR CODE HERE]

    user_input = input("\nWhat do you want to do next?\n You can \n- Quit the chatbot\n- Perform a health check\n- Perform a stylistic analysis of your speech\n")

    next_state = get_state_choice(user_input)

    return next_state

# AISHA ADDED IN PROJECT PART 4:
# In order to keep the check_next_state function visually as clean as simple as it is, I decided to separate the process
# of analyzing the actual user input separately.
def get_state_choice (user_input):
    # Just to make the processing of the user input easier for the regular expression library's methods,
    # I decided to convert it to lowercase
    user_input = user_input.lower()

    # Out of a habit, the user might want to choose an option as if it was an ordered list
    # options of an ordered list include alphabetic and numerical one, so I am accounting for both
    # Furthermore, they might want to quit and phrase that as "I am done"/"Finished"/"End", so i will try to account for that too
    if (re.search(r".*(\bend|\bfinish|\bexit|\bdone|\bquit|\ba$|\b1$|\bi$).*", user_input) != None):
        return "quit"
    elif (re.search(r".*(\bhealth|\bcheck|\bb$|\b2$|\bii$).*", user_input) != None):
        return "health_check"
    elif (re.search(r".*(stylistic|style|analysis|speech|\bc\b|\b3\b|\biii\b).*", user_input) != None):
        return "stylistic_analysis"
    else:
        return "invalid_state"




# ***** New in Project Part 4! *****
# Function: run_chatbot(model):
# model: A trained classification model
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> health_check_state() (OUT STATE)
# health_check_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time health_check_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times health_check_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> health_check_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, word2vec):

    first_time = True
    next_state = welcome_state()

    while (next_state != "quit"):

        if next_state == "get_info":
            next_state = get_info_state()

        elif next_state == "health_check":
            next_state = health_check_state(model, word2vec, first_time)
            first_time = False
        elif next_state == "stylistic_analysis":
            next_state = stylistic_analysis_state()

        elif next_state == "check_next_state":
            next_state = check_next_state()
        else:
            print("I did not quite understand what you meant by that. Let's try again?\n")
            next_state = check_next_state()

    print("Thanks for checking out my bot! Have a great day!")
    return


if __name__ == "__main__":
    lexicon, labels = load_as_list("dataset.csv")

    # Loading the Word2Vec representations so that we can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiating and training the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, lexicon, labels)
    svm = train_model(svm, word2vec, lexicon, labels)
    mlp = train_model(mlp, word2vec, lexicon, labels)

    # Testing the machine learning models to see how they perform on the small
    # test set provided.  Write a classification report to a CSV file with this
    # information.
    test_data, test_labels = load_as_list("test.csv")

    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w")
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_data, test_labels)
        if models[i] == None: # Models will be null if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    # I replaced MLP that was chosen by professor with my best performing model (logistic regression model I trained earlier)
    run_chatbot(logistic, word2vec)
