
from typing import Sequence, Any
import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.start = start
        self.vocab = vocab
        self.obj_to_idx = {obj: idx for idx, obj in enumerate(vocab, start=start)}
        self.idx_to_obj = {idx: obj for obj, idx in self.obj_to_idx.items()}

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.
        """
        return np.array([self.obj_to_idx.get(obj, self.start - 1) for obj in object_seq])

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.
        """
        max_len = max(len(seq) for seq in object_seq_seq)
        index_matrix = np.full((len(object_seq_seq), max_len), fill_value=self.start - 1)
        for i, seq in enumerate(object_seq_seq):
            index_matrix[i, :len(seq)] = self.objects_to_indexes(seq)
        return index_matrix

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.
        """
        binary_vector = np.zeros(len(self.vocab), dtype=int)
        for obj in object_seq:
            idx = self.obj_to_idx.get(obj)
            if idx is not None:
                binary_vector[idx - self.start] = 1
        return binary_vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.
        """
        binary_matrix = np.zeros((len(object_seq_seq), len(self.vocab)), dtype=int)
        for i, seq in enumerate(object_seq_seq):
            binary_matrix[i] = self.objects_to_binary_vector(seq)
        return binary_matrix

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.
        """
        return [self.idx_to_obj[idx] for idx in index_vector if idx in self.idx_to_obj]

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.
        """
        return [self.indexes_to_objects(row) for row in index_matrix]

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.
        """
        return [self.idx_to_obj[idx + self.start] for idx in np.where(vector == 1)[0]]

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.
        """
        return [self.binary_vector_to_objects(row) for row in binary_matrix]
