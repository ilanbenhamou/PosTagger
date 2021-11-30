import numpy as np
from nltk.corpus import brown
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import re


def load_data():
    """
    load the brown dataset and split it into trainning and test set
    :return: training set, test set
    """
    news_text = brown.tagged_sents(categories='news')
    # keeping only the suffix tag when +/- appears
    news_text_update = [[(word, re.split("[+-]", tag)[0]) for (word, tag) in sentence] for sentence in news_text]
    # splitting to train/test set - 90% train, 10 % test
    index = int(len(news_text) * 0.9)
    train_set = news_text_update[:index]
    test_set = news_text_update[index:]
    return train_set, test_set


class PosTagger:
    """
    Abstract class for Pos Tagger model
    """
    def __init__(self):
        self._cfdist_em = defaultdict(lambda: Counter())  # conditional frequency dic for emission probability
        self._cfdist_trans = defaultdict(lambda: Counter())  # conditional frequency dic for transition probability
        self._unique_words = Counter()
        self._unseen_word_tag = "NN"

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y, split=False):
        """
        Calculates accuracy between model prediction from X and GT y
        X and y should have the same shape
        :param X:  array of words
        :param y: array of corresponding tags
        :param split: if true return total accuracy, known words accuracy, unknown words accuracy
                 (Unknown words are words that appear in the test set but not in the training set.)
        :return: total accuracy
        """
        y = np.asarray(y)
        X = np.asarray(X)

        # making prediction
        y_predict = self.predict(X)

        if split:
            return self._split_accuracy(X, y, y_predict)

        else:
            return self._accuracy(y_predict, y)


    def _split_accuracy(self,X, y, pred):
        """
        :param X: array of words
        :param y: array of  GT corresponding tags
        :param pred: prediction tags
        :return: total accuracy, known words accuracy, unknown words accuracy
        """
        acc_tot = self._accuracy(pred, y)
        mask = np.isin(X, list(self._unique_words))
        acc_in = self._accuracy(pred[mask], y[mask])
        acc_not_in = self._accuracy(pred[~mask], y[~mask])
        return np.array([acc_tot, acc_in, acc_not_in])

    @classmethod
    def _accuracy(cls, y, pred):
        """
        :param y: 1d array or list - ground truth tag
        :param pred: 1d array or list - prediction tag
        :return: accuracy score
        """
        y = np.asarray(y)
        pred = np.asarray(pred)
        if len(y) == 0:
            return 1
        return np.sum(y == pred) / len(pred)


class BasicTagger(PosTagger):
    """
    Basic TAG predictor model which computes for each word the tag that maximizes p(tag|word), based
    on the maximum likelihood estimation. The most likely tag of all the unknown
    words is “NN”.
    """

    def __init__(self):
        PosTagger.__init__(self)

    def fit(self, X, y):
        """
        fit the given training set
        :param X: 1-d array of words
        :param y: 1-d arrays of corresponding tags
        """
        for w, t in zip(X, y):
            self._unique_words[w] += 1
            self._cfdist_em[w][t] += 1

    def predict(self, X):
        """
        predict taf for given words
        :param flatten:
        :param X: 1-d array of words
        :return: 1-d array with the resulting tag
        """
        y_pred = [max(self._cfdist_em[word], key=self._cfdist_em[word].get) if word in self._cfdist_em else
                  self._unseen_word_tag for word in X]
        return np.array(y_pred)


class HMM(PosTagger):
    """
    Bigram Hidden Markov Model POS Tagger using the maximum likelihood estimation to compute the transition
    and emission probabilities and Viterbi algorithm to predict best sequence of tags
    Use PseudoWords for unknown words if pw_cat and pw_func is provided
    """

    def __init__(self, smoothing=0., pw_cat=None, pw_func=None):
        """
        :param smoothing: aplace smoothing constant - default zero
        :param pw_cat: list of pseudo_words categories
        :param pw_func: list of pseudo_words function matching pw_cat
        """

        PosTagger.__init__(self)
        self._smoothing = smoothing
        self._start_tag = "**START**"
        self._label = None  # array which assign index to tag
        self._tm = None  # transition matrix
        self._pw_flag = False
        if pw_cat and pw_func and len(pw_cat) == len(pw_func):  # using Pseudo Words
            self._pw = PseudoWords(pw_cat, pw_func)
            self._pw_flag = True
            self._low_threshold = 3

    def _count_words(self, X):
        """
        count unique words and initialize a list with all low frequency word if using pseudowords
        :param X: 2-d array of words [[w1, w2,...],[w1,w2,..],..] each row represents a sentence
        """
        for sentences in X:
            for w in sentences:
                self._unique_words[w] += 1

    def fit(self, X, y):
        """
        fit the model to the given training set (X and y should be the same length)
        :param X: 2-d array of words [[w1, w2,...],[w1,w2,..],..] each row represents a sentence
        :param y: 2-d array of tag [[t1,t2,..],[t1,t2,..],..] each row represents a sentence
        """

        self._count_words(X)
        for w_sent, t_sent in zip(X, y):
            prev_tag = self._start_tag
            for word, tag in zip(w_sent, t_sent):
                self._cfdist_trans[prev_tag][tag] += 1
                if self._pw_flag and self._unique_words[word] < self._low_threshold:  # using pseudo words
                    self._cfdist_em[tag][self._pw.transform(word)] += 1
                else:
                    self._cfdist_em[tag][word] += 1
                prev_tag = tag

        # assign indexes to tags for building the transition matrix (START_TAG will have index 0)
        self._label = [self._start_tag] + list(self._cfdist_em.keys())

        # building Transition Matrix
        n = len(self._label)  # number of unique tags
        self._tm = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                self._tm[i, j] = self.transition_prob(self._label[i], self._label[j])

    def emission_prob(self, word, tag):
        """
        calculates the emission probability p(word| tag)
        :param word: str
        :param tag: str
        :return: probability p(word| tag)
        """
        if tag not in self._cfdist_em:
            return 0

        # using pseudo_words
        elif self._pw_flag and (self._unique_words[word] < self._low_threshold):
            word = self._pw.transform(word)

        # choosing an arbitrary tag for unknown words when there is no smoothing
        elif not self._smoothing and word not in self._unique_words and tag == self._unseen_word_tag:
            return 1

        return (self._cfdist_em[tag][word] + self._smoothing) / (sum(self._cfdist_em[tag].values()) + self._smoothing *
                                                                 len(self._unique_words))

    def transition_prob(self, tag, prev_tag):
        """
        calculates the transition probability p(tag|prev_tag)
        :param tag: str
        :param prev_tag: str
        :return:  p(tag|prev_tag)
        """
        if prev_tag not in self._cfdist_trans:
            return 0
        return self._cfdist_trans[prev_tag][tag] / sum(self._cfdist_trans[prev_tag].values())

    def __viterbi(self, sentence):
        """
        viterbi algorithm to predict sequence of tags
        :param sentence: 1d array - [w1, w2, w3,..] sentence of words
        :return: 1d array [t1, t2, t3] optimal sequence of tags corresponding to the given words
        """
        n = len(sentence) + 1
        num_of_tag = len(self._cfdist_trans)
        prob_mat = np.zeros((n, num_of_tag))  # used to keep all best sequences probabilities at all steps
        path_mat = np.zeros((n - 1, num_of_tag))  # used for backtracking the best sequence
        prob_mat[0, 0] = 1  # always starting from the START tag which have index 0

        # filling the dynamic array
        for step in range(1, n):
            for indx_tag in range(num_of_tag):
                tmp_prob = prob_mat[step - 1, :] * self._tm[indx_tag, :]
                em = self.emission_prob(sentence[step - 1], self._label[indx_tag])
                prob_mat[step, indx_tag] = np.max(tmp_prob) * em
                path_mat[step - 1, indx_tag] = np.argmax(tmp_prob)

        # Finding optimal sequence
        opt_seq = np.zeros(n)
        opt_seq[-1] = np.argmax(prob_mat[-1, :])

        for step in range(n - 2, 0, -1):
            opt_seq[step] = path_mat[step, int(opt_seq[step + 1])]

        return [self._label[int(i)] for i in opt_seq[1:]]  # index to tag

    def predict(self, X):
        """
        predict sequence of tags for sentences by using the viterbi algorithm
        :param X: [[word, word..],[word, word,..],..]
        :return: y - [[tag, tag..],[tag, tag,..],..] array of list
        """
        y = []
        for sentence in X:
            y.append(self.__viterbi(sentence))
        return np.array(y)

    def score(self, X, y, split=False):
        """
        :param X: [[word, word..],[word, word,..],..]
        :param y: [[tag, tag..],[tag, tag,..],..] array of list
        :param split: if true return total accuracy, known words accuracy, unknown words accuracy
                 (Unknown words are words that appear in the test set but not in the training set.)
        :return: accuracy score
        """
        y_predict = self.predict(X)
        X = np.concatenate(X).ravel()
        y = np.concatenate(y).ravel()
        y_predict = np.concatenate(y_predict).ravel()
        return PosTagger._split_accuracy(self, X, y, y_predict)

class PseudoWords:
    """
    Object of this class can trainsform words into pseudo-words or categories.
    """
    def __init__(self, cat_name, cat_func):
        """
        cat_name and cat_func should have the same length
        :param cat_name: list of string
        :param cat_func: list of function which correspond to the categories name.
                        All functions should return a boolean value and take one string as argument
        """
        self._categories = cat_name
        self._cat_func = cat_func

    def transform(self, words):
        """
        transform words to one of the categories in cat_name by applying the cat_func functions
        :param words: list of str or str
        :return: str or list of str(same len as input) of the first matching categories for all words
        """
        result = []
        if isinstance(words, str):
            words = [words]
        for w in words:
            for cat_name, cat_func in zip(self._categories, self._cat_func):
                if cat_func(w):
                    if len(words) == 1:
                        return cat_name
                    result.append(cat_name)
                    break
                    
        return np.array(result)


# pseudo words functions
def first_cap(w):
    return w[0].isupper()


def compound_word(w):
    return re.match("[a-zA-Z]+-[a-zA-Z]+", w) is not None


def date(w):
    return re.match("\d+/\d+/\d*", w) is not None or re.match("\d+-\d+-\d*", w) is not None


def money(w):
    return "$" in w or re.match("\d+[,.]\d*", w) is not None


def other(w):
    return True



