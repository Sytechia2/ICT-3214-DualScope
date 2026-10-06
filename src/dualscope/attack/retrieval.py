"""Keyword retrieval over the ATT&CK technique catalogue (Task 6.1).

A TF-IDF index over each technique's name, description and data components,
ranked by cosine similarity. It needs no model download or GPU, and the same
catalogue and query always give the same ranking. Results below ``min_score``
are dropped, so a weak query can return nothing instead of a forced match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from dualscope.attack.catalog import Technique, TechniqueCatalog

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "data" / "reference" / "attack" / "enterprise_techniques.json"

# ATT&CK data components an authentication log such as LANL auth.txt can show.
AUTH_DATA_COMPONENTS = frozenset({
    "Active Directory Credential Request",
    "Logon Session Creation",
    "Logon Session Metadata",
    "User Account Authentication",
})
DEFAULT_MIN_SCORE = 0.10
NAME_WEIGHT = 3  # the name is repeated so it counts more than one description word


@dataclass(frozen=True)
class RetrievedTechnique:
    technique_id: str
    name: str
    score: float
    url: str
    tactics: tuple[str, ...]
    data_components: tuple[str, ...]
    auth_observable: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "score": round(self.score, 4),
            "url": self.url,
            "tactics": list(self.tactics),
            "data_components": list(self.data_components),
            "auth_observable": self.auth_observable,
            "description": self.description,
        }


def auth_observable(technique: Technique) -> bool:
    return bool(AUTH_DATA_COMPONENTS.intersection(technique.data_components))


def _document(technique: Technique) -> str:
    return " ".join([*[technique.name] * NAME_WEIGHT, technique.description, *technique.data_components])


class TechniqueRetriever:
    def __init__(self, catalog: TechniqueCatalog):
        self.catalog = catalog
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform([_document(t) for t in catalog.techniques])
        self._observable = np.array([auth_observable(t) for t in catalog.techniques])

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CATALOG) -> TechniqueRetriever:
        return cls(TechniqueCatalog.load(path))

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        auth_observable_only: bool = False,
    ) -> list[RetrievedTechnique]:
        """Top ``k`` techniques for ``query`` scoring at least ``min_score``.

        With ``auth_observable_only``, techniques whose ATT&CK data components
        include nothing an authentication log can show are skipped.
        """
        if not query.strip():
            return []
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        if auth_observable_only:
            scores = np.where(self._observable, scores, -1.0)
        # Ties break by catalogue order (technique ID), so results are stable.
        order = np.lexsort((np.arange(len(scores)), -scores))
        results = []
        for index in order[:k]:
            if scores[index] < min_score:
                break
            technique = self.catalog.techniques[index]
            results.append(RetrievedTechnique(
                technique_id=technique.technique_id,
                name=technique.name,
                score=float(scores[index]),
                url=technique.url,
                tactics=technique.tactics,
                data_components=technique.data_components,
                auth_observable=bool(self._observable[index]),
                description=technique.description,
            ))
        return results
