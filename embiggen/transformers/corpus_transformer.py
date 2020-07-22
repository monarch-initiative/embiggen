"""Module offers basic Corpus Transformer object, a simple class to tekenize textual corpuses."""
import string
from typing import Dict, List

import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer   # pylint: disable=import-error
from tqdm.auto import tqdm


class CorpusTransformer:
    """Simple class to tekenize textual corpuses."""

    def __init__(
        self,
        synonyms: Dict = None,
        language: str = "english",
        apply_stemming: bool = False,
        extend_synonyms: bool = True
    ):
        """Create new CorpusTransformer object.

        This is a GENERIC text tokenizer and is only useful for basic examples
        as in any advanced settings there will be need for a custom tokenizer.

        Parameters
        ----------------------------
        synonyms: Dict = None,
            The synonyms to use.
        language: str = "english",
            The language for the stopwords.
        apply_stemming: bool = False,
            Wethever to apply or not a stemming procedure, which
            by default is disabled.
            The algorithm used is a Porter Stemmer.
        extend_synonyms: bool = True,
            Wethever to automatically extend the synonyms using wordnet.
        """
        self._synonyms = {} if synonyms is None else synonyms
        self._stopwords = set(stopwords.words(
            language)) | set(string.punctuation)
        self._stemmer = PorterStemmer()
        self._apply_stemming = apply_stemming
        self._extend_synonyms = extend_synonyms
        self._tokenizer = None

    def get_synonym(self, word: str) -> str:
        """Return the synonym of the given word, if available.

        Parameters
        ----------------------------
        word: str,
            The word whose synonym is to be found.

        Returns
        ----------------------------
        The given word synonym.
        """
        if word not in self._synonyms:
            if not self._extend_synonyms:
                return word
            possible_synonyms = wn.synsets(word)
            if possible_synonyms:
                word_synonyms = [
                    w.lower()
                    for w in possible_synonyms[0].lemma_names()
                ]
                self._synonyms[word] = word_synonyms[0]
                self._synonyms[word_synonyms[0]] = word_synonyms[0]
            else:
                self._synonyms[word] = word

        return self._synonyms[word]

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
                    synonym = self.get_synonym(word)
                    if self._apply_stemming:
                        synonym = self._stemmer.stem(synonym)
                    counter[synonym] = counter.setdefault(synonym, 0) + 1
                    tokens.append(synonym)
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
        _, counts = self.tokenize(texts, True, verbose)

        self._stopwords |= {
            word
            for word, count in counts.items()
            if count <= min_count
        }

        self._tokenizer = Tokenizer()
        self._tokenizer.fit_on_texts((
            " ".join(tokens)
            for tokens in self.tokenize(texts, False, verbose)
        ))

    @property
    def vocabulary_size(self) -> int:
        """Return number of different terms."""
        return len(self._tokenizer.word_counts)

    def reverse_transform(self, sequences: List[List[int]]) -> List[str]:
        """Reverse the sequence to texts.

        Parameters
        ------------------------
        sequences: List[List[int]],
            The sequences to counter transform.

        Returns
        ------------------------
        The texts created from the given sequences.
        """
        return self._tokenizer.sequences_to_texts(sequences)

    def get_word_id(self, word: str) -> int:
        """Get the given words IDs.

        Parameters
        ------------------------
        word: int
            The word whose ID is to be retrieved.

        Returns
        ------------------------
        The word numeric ID.
        """
        return self._tokenizer.word_index[word]

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
