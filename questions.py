import nltk
import sys
import os
import string
import math
#from nltk.tokenize import word_tokenize

from yaml.events import DocumentEndEvent
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            with open(file_path) as f:
                lines = f.read()

            dictionary.update({filename: lines})
    
    return dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords and lemmatizing the words to only include
    the lemma.
    """
    
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    words = nltk.word_tokenize(document.lower())
    wordnet_lemmatizer = WordNetLemmatizer()

    lemmanized_words = set()
    delete_word = set()
    stopwords = set(nltk.corpus.stopwords.words("english"))
    punctuation = set(string.punctuation)

    for token, tag in pos_tag(words):
        lemma = wordnet_lemmatizer.lemmatize(token, tag_map[tag[0]])
        #print(token, "=>", lemma)
        lemmanized_words.add(lemma)
    
    #print(lemmanized_words)
    
    for word in lemmanized_words:
        if word in stopwords or word in punctuation:
            delete_word.add(word)

    #return tokenized and lemmanized words that are not excluded
    return [word for word in lemmanized_words if word not in delete_word]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    #print("Calculating inverse document frequencies...")

    # create and empty dict to map words to idfs and a word_list set for each doc
    idfs = dict()
    word_list = set()

    for key, values in documents.items():
        word_list.update(set(values))

    for word in word_list:

        # natural logarithm of doc # divided by docs in which the word appears
        f = sum(word in documents[key] for key in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    #print("Calculating TF-IDFs for top files...")

    # Calculate the TF-IDFs for input, ensure it matches query
    tfidfs = []

    for name in files:
        tfidf = 0

        # check for unknown words in query to avoid keyErrors
        try:
            for word in query:

                # calculate tfidf by multiplying tf with idf for word
                tfidf += (files[name].count(word) * idfs[word])
        
            tfidfs.append((name, tfidf))

            # Sort and get top n TF-IDFs for each file
            tfidfs.sort(key=lambda x: x[1], reverse=True)
    
        except KeyError:
            print("Unknown term in your query.")
            sys.exit()

    return [x[0] for x in tfidfs[:n]]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    #print("Computing top sentences...")

    top_sentences = []

    # Calculate IDFs based on words in sentence that are in query
    for sentence, words in sentences.items():
        word_count = 0
        sentence_idfs = 0

        # only check for words in query to avoid long calculations
        for word in query:
            
            # check that word exists in sentence and calucalte idfs
            if word in words:
                word_count += 1
                sentence_idfs += idfs[word]

        query_term_density = word_count / len(words)
        top_sentences.append((sentence, sentence_idfs, query_term_density))

    # Order sentences with the best match first
    top_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return [x[0] for x in top_sentences[:n]]


if __name__ == "__main__":
    main()
