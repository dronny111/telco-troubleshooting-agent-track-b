#!/usr/bin/env python3
"""Text feature extractors for the Track A ranker."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """Sparse BM25 text vectorizer with a sklearn-like interface.

    This is intentionally small and pickle-friendly. It mirrors the
    `fit_transform`/`transform` behavior used by TfidfVectorizer so ranker
    bundles can swap TF-IDF and BM25 without changing scoring code.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 2,
        max_features: int | None = None,
        strip_accents: str | None = "unicode",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.strip_accents = strip_accents
        self.k1 = k1
        self.b = b

    def fit(self, raw_documents, y=None):
        self.count_vectorizer_ = CountVectorizer(
            lowercase=self.lowercase,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            strip_accents=self.strip_accents,
        )
        counts = self.count_vectorizer_.fit_transform(raw_documents)
        self._fit_bm25_stats(counts)
        return self

    def fit_transform(self, raw_documents, y=None):
        self.count_vectorizer_ = CountVectorizer(
            lowercase=self.lowercase,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            strip_accents=self.strip_accents,
        )
        counts = self.count_vectorizer_.fit_transform(raw_documents)
        self._fit_bm25_stats(counts)
        return self._bm25_transform(counts)

    def transform(self, raw_documents):
        counts = self.count_vectorizer_.transform(raw_documents)
        return self._bm25_transform(counts)

    def get_feature_names_out(self, input_features=None):
        return self.count_vectorizer_.get_feature_names_out(input_features)

    def _fit_bm25_stats(self, counts) -> None:
        counts = counts.tocsr()
        n_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf_ = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        self.avgdl_ = float(doc_len.mean()) if doc_len.size else 0.0

    def _bm25_transform(self, counts):
        counts = counts.tocsr(copy=True).astype(np.float64)
        if counts.nnz == 0:
            return counts
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = self.avgdl_ if self.avgdl_ > 0 else 1.0
        row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        norm = self.k1 * (1.0 - self.b + self.b * doc_len[row_ids] / avgdl)
        counts.data = (
            self.idf_[counts.indices]
            * counts.data
            * (self.k1 + 1.0)
            / (counts.data + norm)
        )
        return sparse.csr_matrix(counts)
