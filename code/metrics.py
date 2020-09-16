import os
from multiprocessing import Pool
import random
import pdb
import numpy as np
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import nist_score, bleu_score

try: 
    from multiprocessing import cpu_count
except: 
    from os import cpu_count

class Metrics(object):
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_score(self):
        pass

class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, num_real_sentences=500, num_fake_sentences=10000, smoothing_method=None, chunk_size=10):
        super(Bleu, self).__init__()
        self.name = 'Bleu'
        self.test_data = test_text
        self.smoothing_method = smoothing_method
        self.real_data = real_text
        self.gram = gram
        self.sample_size = num_real_sentences
        self.reference = None
        self.is_first = True
        self.num_sentences = num_fake_sentences
        self.CHUNK_SIZE = chunk_size


    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    # fetch REAL DATA
    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            random.shuffle(reference)
            reference = reference[0:self.sample_size]
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        raise Exception('make sure you call BLEU paralell')
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                bleu.append(bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=self.smoothing_method))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=self.smoothing_method)

    def calc_bleu_list(self, reference, hypotheses, weight):
        """
        Args:
        - Reference: list(list(str)) -- a corpus of list of tokenized reference sentences from the corpus 
        - Hypotheses: list(list(str)) -- a list of reference sentences from the corpus 
        """
        scores = []
        for hypothesis in hypotheses:
            scores.append(bleu_score.sentence_bleu(reference, hypothesis, weight,
                           smoothing_function=self.smoothing_method))
        return scores 

    def get_bleu_fast(self):
        reference = self.get_reference()
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        maxx = self.num_sentences
        with open(self.test_data) as test_data:
            all_hypotheses = []
            shuffled_hypotheses = test_data.readlines()
            random.shuffle(shuffled_hypotheses)

            for i, hypothesis in enumerate(shuffled_hypotheses):
                # print(len(reference), len(hypothesis))
                # print(hypothesis)
                # input("Waiting!")
                #print('i : {}'.format(i))
                hypothesis = nltk.word_tokenize(hypothesis)
                if len(hypothesis)==0:
                    continue
                if i > maxx : break
                # result.append(pool.apply_async(self.calc_bleu_list, args=(reference, hypotheses, weight)))
                all_hypotheses.append(hypothesis)

            print(f"Now computing the {len(all_hypotheses)} hypotheses!")
            for idx in range(0, len(all_hypotheses), self.CHUNK_SIZE):
                hypotheses = all_hypotheses[idx:idx+self.CHUNK_SIZE] 
                result.append(pool.apply_async(self.calc_bleu_list, args=(reference, hypotheses, weight)))

        score = 0.0
        cnt = 0
        scores = []
        for it, i in enumerate(result):
            #print('i : {}'.format(it))
            # score += i.get()
            # cnt += 1
            scores.extend(i.get())
        pool.close()
        pool.join()
        return sum(scores) / len(scores) 


class SelfBleu(Metrics):
    def __init__(self, test_text='', real_text="", gram=3, model_path='', num_real_sentences=500, num_fake_sentences=500, smoothing_method=None, chunk_size=10):
        super(SelfBleu, self).__init__()
        self.name = 'Self-Bleu'
        # if (num_real_sentences!=num_fake_sentences):
            # raise Exception("Number of fake sentences does not equal number of real ones!")
        self.test_data = test_text
        self.gram = gram
        self.smoothing_method = smoothing_method
        self.sample_size = num_real_sentences 
        self.reference = None
        self.is_first = True
        self.CHUNK_SIZE = chunk_size

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as f:
                real_data = f.readlines() 
                random.shuffle(real_data)
                real_data = real_data[:self.sample_size]
                idx = 0
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=self.smoothing_method))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=self.smoothing_method)

    def get_bleu_fast(self):
        reference = self.get_reference()
        random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def calc_bleu_list(self, chunk, weight):
        scores = []
        for item in chunk:
            reference, hypothesis = item
            scores.append(bleu_score.sentence_bleu(reference, hypothesis, weight,
                       smoothing_function=self.smoothing_method))
        return scores

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(cpu_count())
        result = list()
        sentence_num = len(reference)
        preprocessed = []
        for index in range(sentence_num):
            #genious:
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            preprocessed.append((other, hypothesis))

        random.shuffle(preprocessed)
        for idx in range(0, len(preprocessed), self.CHUNK_SIZE):
            chunk = preprocessed[idx:idx+self.CHUNK_SIZE]
            result.append(pool.apply_async(self.calc_bleu_list, args=(chunk, weight)))

        scores = []
        for it, i in enumerate(result):
            #print('i : {}'.format(it))
            # score += i.get()
            # cnt += 1
            scores.extend(i.get())
        pool.close()
        pool.join()
        return sum(scores) / len(scores) 

class Nist(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, num_real_sentences=500, num_fake_sentences=10000):
        super(Nist, self).__init__()
        raise Exception("We don't use this anymore!")
        self.name = 'NIST'
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = num_real_sentences
        self.reference = None
        self.is_first = True
        self.num_sentences = num_fake_sentences

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_nist_fast()
        return self.get_nist()

    # fetch REAL DATA
    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_nist(self):
        # raise Exception('make sure you call NIST paralell')
        ngram = self.gram
        nist = list()
        reference = self.get_reference()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                if len(hypothesis)==0:
                    continue
                nist.append(self.calc_nist(reference, hypothesis, gram=ngram)) 
        return sum(nist) / len(nist)

    def calc_nist(self, reference, hypothesis, gram=5):
        print("Type:", type(reference[0][0]))
        print("Lengths:", len(reference), len(hypothesis))
        print("Hypothesis 2:", hypothesis)
        return nist_score.sentence_nist(reference, hypothesis, n=gram) 

    def get_nist_fast(self):
        reference = self.get_reference()
        reference = reference[0:self.sample_size]
        return self.get_nist_parallel(reference=reference)

    def get_nist_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        pool = Pool(cpu_count())
        result = list()
        maxx = self.num_sentences
        with open(self.test_data) as test_data:
            for i, hypothesis in enumerate(test_data):
                #print('i : {}'.format(i))
                # hypothesis = nltk.word_tokenize(hypothesis)
                hypothesis = nltk.word_tokenize(hypothesis)
                if len(hypothesis)==0:
                    continue
                result.append(pool.apply_async(self.calc_nist, args=(reference, hypothesis, ngram)))
                if i > maxx : break
        score = 0.0
        cnt = 0
        for it, i in enumerate(result):
            #print('i : {}'.format(it))
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt


class SelfNist(Metrics):
    def __init__(self, test_text='', real_text="", gram=3, model_path='', num_real_sentences=500, num_fake_sentences=500):
        super(SelfNist, self).__init__()
        if num_real_sentences!=num_fake_sentences:
            raise Exception("Please ensure that the number of real sentences is the same as generated ones!")
        self.name = 'Self-NIST'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = num_real_sentences 
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_nist_fast()
        return self.get_nist_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    if len(text)==0:
                        continue
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_nist(self):
        ngram = self.gram
        nist = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                nist.append(nist_score.sentence_nist(reference, hypothesis, n=ngram))
        return sum(nist) / len(nist)

    def calc_nist(self, reference, hypothesis, gram):
        return nist_score.sentence_nist(reference, hypothesis, n=gram) 

    def get_nist_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_nist_parallel(reference=reference)

    def get_nist_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        pool = Pool(cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            #genious:
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_nist, args=(other, hypothesis, ngram)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

