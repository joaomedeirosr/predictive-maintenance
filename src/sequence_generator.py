import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Generator


class SequenceGenerator:
    """
    Responsável por gerar as sequências de entrada (X)
    e os rótulos (y) para modelos de séries temporais.
    """

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def generate_sequences(
        self,
        feature_df: pd.DataFrame,
        seq_cols: list[str]
    ) -> Generator[NDArray[np.float32], None, None]:
        """
        Gera as sequências de um único motor.
        """

        feature_array = feature_df[seq_cols].values
        num_elements = feature_array.shape[0]

        if num_elements <= self.sequence_length:
            return

        for start, stop in zip(
            range(0, num_elements - self.sequence_length),
            range(self.sequence_length, num_elements)
        ):
            yield feature_array[start:stop].astype(np.float32)

    def create_sequence_dataset(
        self,
        df: pd.DataFrame,
        seq_cols: list[str]
    ) -> NDArray[np.float32]:
        """
        Gera todas as sequências do dataset.
        """

        sequences: list[NDArray[np.float32]] = []

        for engine_id in df["id"].unique():

            engine_sequences = list(
                self.generate_sequences(
                    df[df["id"] == engine_id],
                    seq_cols
                )
            )

            if engine_sequences:
                sequences.append(np.array(engine_sequences))

        return np.concatenate(sequences, axis=0).astype(np.float32)

    def generate_labels(
        self,
        label_df: pd.DataFrame,
        label: list[str]
    ) -> NDArray[np.float32]:
        """
        Gera os labels de um único motor.
        """

        label_array = label_df[label].values
        num_elements = label_array.shape[0]

        if num_elements <= self.sequence_length:
            return np.empty((0, len(label)), dtype=np.float32)

        return label_array[self.sequence_length:].astype(np.float32)

    def create_label_dataset(
        self,
        df: pd.DataFrame,
        label: list[str]
    ) -> NDArray[np.float32]:
        """
        Gera todos os labels do dataset.
        """

        labels: list[NDArray[np.float32]] = []

        for engine_id in df["id"].unique():

            engine_labels = self.generate_labels(
                df[df["id"] == engine_id],
                label
            )

            if len(engine_labels) > 0:
                labels.append(engine_labels)

        return np.concatenate(labels, axis=0).astype(np.float32)

    def create_dataset(
        self,
        df: pd.DataFrame,
        seq_cols: list[str],
        label: list[str]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Gera simultaneamente X e y.
        """

        X = self.create_sequence_dataset(df, seq_cols)
        y = self.create_label_dataset(df, label)

        return X, y