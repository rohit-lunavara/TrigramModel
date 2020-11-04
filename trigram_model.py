import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
Name - Rohit Lunavara
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    # Set prefix and postfix
    prefix = "START"
    postfix = "STOP"
    # Append 1 prefix regardless of value of n
    sequence.insert(0, prefix)
    for i in range(n - 2) :
        sequence.insert(0, prefix)
    sequence.append(postfix)

    # Check if 1 <= n <= len(sequence)
    if (n > len(sequence)) :
        print(sequence)
    assert 1 <= n
    assert n <= len(sequence)

    # Create n-gram sequence
    n_gram_sequence = []
    for i in range(n, len(sequence) + 1) :
        temp_sequence = []
        for j in range(i - n, i) :
            temp_sequence.append(sequence[j])
        n_gram_sequence.append(tuple(temp_sequence))
    return n_gram_sequence

class TrigramModel(object):
    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        self.total_word_count = 0
        self.total_sentence_count = 0

        # For each sentence, generate ngram_counts
        for sentence in corpus :
            # 2 extra START tokens and 1 extra STOP token
            self.total_word_count += len(sentence) + 3
            self.total_sentence_count += 1
            for unigram in get_ngrams(sentence, 1) :
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, 2) :
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3) :
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """

        assert len(trigram) == 3
        bigram = trigram[:2]
        count_bigram = self.bigramcounts[bigram]
        count_trigram = self.trigramcounts[trigram]
        if count_bigram == 0 :
            return 1 / self.total_sentence_count
        else :
            return (count_trigram / count_bigram)

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """

        assert len(bigram) == 2
        unigram = bigram[:1]
        count_unigram = self.unigramcounts[unigram]
        count_bigram = self.bigramcounts[bigram]
        if count_unigram == 0 :
            return 1 / self.total_sentence_count
        else :
            return (count_bigram / count_unigram)

    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """

        assert len(unigram) == 1
        count_unigram = self.unigramcounts[unigram]
        if self.total_word_count == 0 :
            return 1 / self.total_sentence_count
        else :
            return (count_unigram / self.total_word_count)

    def generate_sentence(self,t=20): 
        """
        TODO :
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = ""
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        assert len(trigram) == 3

        unigram = trigram[:1]
        bigram = trigram[:2]
        return (lambda1 * self.raw_trigram_probability(trigram) 
        + lambda2 * self.raw_bigram_probability(bigram) 
        + lambda3 * self.raw_unigram_probability(unigram))

    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        sum_logprob = 0
        trigrams = get_ngrams(sentence, 3)
        for trigram in trigrams :
            sum_logprob += math.log2(self.smoothed_trigram_probability(trigram))

        return sum_logprob

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        l = 0
        total_word_count = 0
        for sentence in corpus :
            l += self.sentence_logprob(sentence)
            # 2 extra START tokens and 1 extra STOP token
            total_word_count += len(sentence)
        l /= total_word_count
        return math.pow(2, -l)


def essay_scoring_experiment(training_file_high, training_file_low, testdir_high, testdir_low):
    model_high = TrigramModel(training_file_high)
    model_low = TrigramModel(training_file_low)

    total = 0
    correct = 0       

    for f in os.listdir(testdir_high):
        pp_high = model_high.perplexity(corpus_reader(os.path.join(testdir_high, f), model_high.lexicon))
        pp_low = model_low.perplexity(corpus_reader(os.path.join(testdir_high, f), model_low.lexicon))
        # Since actual label is high, if perplexity is lower for high model, then that prediction is correct
        if pp_high <= pp_low :
            correct += 1
        total += 1

    for f in os.listdir(testdir_low):
        pp_high = model_high.perplexity(corpus_reader(os.path.join(testdir_low, f), model_high.lexicon))
        pp_low = model_low.perplexity(corpus_reader(os.path.join(testdir_low, f), model_low.lexicon))
        # Since actual label is low, if perplexity is lower for low model, then that prediction is correct
        if pp_high >= pp_low :
            correct += 1
        total += 1

    return (correct / total)

if __name__ == "__main__":
    # # get_ngrams

    # # Will raise AssertionError
    # # print(get_ngrams(["natural","language","processing"],0))
    # # print(get_ngrams(["natural","language","processing"],4))

    # print(get_ngrams(["natural","language","processing"],1))
    # print(get_ngrams(["natural","language","processing"],2))
    # print(get_ngrams(["natural","language","processing"],3))

    # # count_ngrams

    # # Will raise AssertionError
    # # print(model.raw_trigram_probability(("the")))
    # # print(model.raw_bigram_probability(("START", "START", "the")))
    # # print(model.raw_unigram_probability(("START", "START")))

    # print(model.raw_unigram_probability(("START",)))
    # print(model.raw_unigram_probability(("the",)))
    # print(model.raw_bigram_probability(("START", "START")))
    # print(model.raw_bigram_probability(("START", "the")))
    # print(model.raw_trigram_probability(("START", "START", "the")))

    # # smoothed_trigram_probability

    # # Will raise AssertionError
    # # print(model.smoothed_trigram_probability(("the")))
    # # print(model.smoothed_trigram_probability(("START", "START")))

    # print(model.smoothed_trigram_probability(("START", "START", "the")))
    # print(model.smoothed_trigram_probability(("START", "START", "hello")))
    # print(model.smoothed_trigram_probability(("START", "START", "hi")))

    # # sentence_logprob
    # print(model.sentence_logprob(["the", "natural", "world"]))
    # print(model.sentence_logprob(["the", "beautiful", "world"]))
    # print(model.sentence_logprob(["the", "most", "beautiful"]))

    # # perplexity
    # model = TrigramModel(sys.argv[1])
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)
    # train_corpus = corpus_reader(sys.argv[1], model.lexicon)
    # pp2 = model.perplexity(train_corpus)
    # print(pp2)

    # Essay scoring experiment: 
    acc = essay_scoring_experiment("data/ets_toefl_data/train_high.txt", "data/ets_toefl_data/train_low.txt", "data/ets_toefl_data/test_high", "data/ets_toefl_data/test_low")
    print("Accuracy :", acc)