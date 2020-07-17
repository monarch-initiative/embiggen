from typing import Set, List, Dict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm.auto import tqdm
import string


class CorpusTransformer:

    def __init__(
        self,
        synonims: Dict = None,
        language: str = "english",
        apply_stemming: bool = False,
        extend_synonims: bool = True
    ):
        """Create new CorpusTransformer object.

        Parameters
        ----------------------------
        synonims: Dict = None,
            The synonims to use.
        language: str = "english",
            The language for the stopwords.
        apply_stemming: bool = False,
            Wethever to apply or not a stemming procedure, which
            by default is disabled.
            The algorithm used is a Porter Stemmer.
        extend_synonims: bool = True,
            Wethever to automatically extend the synonims using wordnet.
        """
        self._synonims = {} if synonims is None else synonims
        self._stopwords = set(stopwords.words(
            language)) | set(string.punctuation)
        self._stemmer = PorterStemmer()
        self._apply_stemming = apply_stemming
        self._extend_synonims = extend_synonims
        self._tokenizer = None

    def get_synonim(self, word: str) -> str:
        """Return the synonim of the given word, if available.

        Parameters
        ----------------------------
        word: str,
            The word whose synonim is to be found.

        Returns
        ----------------------------
        The given word synonim.
        """
        if word not in self._synonims:
            if not self._extend_synonims:
                return word
            possible_synonims = wn.synsets(word)
            if possible_synonims:
                word_synonims = [
                    w.lower()
                    for w in possible_synonims[0].lemma_names()
                ]
                for w in word_synonims:
                    self._synonims[w] = word_synonims[0]
                self._synonims[word] = word_synonims[0]
            else:
                self._synonims[word] = word

        return self._synonims[word]

    def tokenize(self, texts: List[str], return_counts: bool = False, verbose: bool = True):
        """Fit model using stemming from given text.

        Parameters
        ----------------------------
        texts: List[str],
            The text to use to fit the transformer.
        return_counts: bool = False,
            Wethever to return the counts of the terms or not.
        verbose: bool = True,
            Wethever to show or not tokenization loading bar.

        Return
        -----------------------------
        Either the tokens or tuple containing the tokens and the counts.
        """
        all_tokens = []
        counter = {}
        for line in tqdm(texts, desc="Tokenizing", disable=not verbose):
            tokens = []
            for word in word_tokenize(line.lower()):
                if word not in self._stopwords:
                    synonim = self.get_synonim(word)
                    if self._apply_stemming:
                        synonim = self._stemmer.stem(synonim)
                    counter[synonim] = counter.setdefault(synonim, 0) + 1
                    tokens.append(synonim)
            all_tokens.append(tokens)

        if return_counts:
            return all_tokens, counter
        return all_tokens

    def fit(self, texts: List[str], min_count: int = 0, verbose: bool = True):
        """Fit the trasformer.

        Parameters
        ----------------------------
        texts: List[str],
            The texts to use for the fitting.
        min_count: int = 0,
            Minimum count to consider the word term.
        verbose: bool = True,
            Wethever to show or not the loading bars.
        """
        tokens, counts = self.tokenize(texts, True, verbose)

        self._stopwords |= {
            word
            for word, count in counts.items()
            if count <= min_count
        }

        self._tokenizer = Tokenizer()
        self._tokenizer.fit_on_texts(tokens)

    @property
    def vocabulary_size(self) -> int:
        """Return number of different terms."""
        return len(self._tokenizer.word_counts)

    def transform(self, texts: List[str], min_length: int = 0, verbose: bool = True) -> np.ndarray:
        """Transform given text.

        Parameters
        --------------------------
        texts: List[str],
            The texts to encode as digits.
        min_length: int = 0,
            Minimum length of the single texts.
        verbose: bool = True,
            Wethever to show or not the loading bar.

        Returns
        --------------------------
        Numpy array with numpy arrays of tokens.
        """
        return np.array([
            np.array(tokens, dtype=np.int64) - 1
            for tokens in self._tokenizer.texts_to_sequences((
                " ".join(tokens)
                for tokens in self.tokenize(texts, verbose=verbose)
                if len(tokens) > min_length
            ))
        ])
