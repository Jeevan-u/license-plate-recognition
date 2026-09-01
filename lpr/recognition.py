"""Character recognition: classify segmented characters with an SVM.

Each segmented character is flattened into a feature vector (its pixel
intensities) and fed to an SVM with an RBF kernel, trained on labeled
character images. This is the "machine learning" part of the pipeline called
out in the tutorial.

The recognizer ships capable of training a fresh model, saving it to disk,
and using it to predict plate text.
"""

import os
import pickle
from typing import List, Optional

import numpy as np

try:
    from sklearn import svm as sklearn_svm
    from sklearn.model_selection import train_test_split
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "char_svm.pkl")
)


class CharacterRecognizer:
    """Recognizes characters using a trained SVM model."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            self._load(model_path)

    @property
    def trained(self) -> bool:
        return self.model is not None

    def _load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def train(self, images: List[np.ndarray], labels: List[str]) -> dict:
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required to train the character recognizer")

        X = np.array([img.flatten().astype(np.float32) for img in images])
        y = np.array(labels)

        clf = sklearn_svm.SVC(kernel="rbf", gamma="scale", C=1.0)
        clf.fit(X, y)
        self.model = clf
        self.save()

        scores = {"train_accuracy": clf.score(X, y)}
        if len(np.unique(y)) > 1 and len(y) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            clf2 = sklearn_svm.SVC(kernel="rbf", gamma="scale", C=1.0)
            clf2.fit(X_train, y_train)
            scores["test_accuracy"] = clf2.score(X_test, y_test)
        return scores

    def predict(self, images: List[np.ndarray]) -> List[str]:
        if self.model is None:
            raise RuntimeError("No trained model available; train or load one first")
        X = np.array([img.flatten().astype(np.float32) for img in images])
        return list(self.model.predict(X))
