"""Visualização dos resultados de treinamento e avaliação do modelo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def save_or_show(save_path: str | Path | None) -> None:
    """Salva o gráfico em arquivo, ou mostra na tela se nenhum caminho for dado."""
    plt.tight_layout()

    if save_path is None:
        plt.show()
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em {path}")


def plot_losses(
    train_losses: list[float],
    test_losses: list[float] | None = None,
    title: str = "Curva de perda por época",
    ylabel: str = "Erro (loss)",
    save_path: str | Path | None = None,
) -> None:
    """Desenha a curva de perda do treino (e do teste, se houver) por época."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Treino")

    if test_losses is not None:
        plt.plot(epochs, test_losses, label="Teste")

    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show(save_path)


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: tuple[str, str] = ("Saudável", "Vai falhar"),
    title: str = "Matriz de confusão",
    save_path: str | Path | None = None,
) -> None:
    """Desenha a matriz de confusão devolvida pelo evaluator."""
    data = np.asarray(matrix)
    meio = data.max() / 2

    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="Blues")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for linha in range(data.shape[0]):
        for coluna in range(data.shape[1]):
            plt.text(
                coluna,
                linha,
                str(data[linha, coluna]),
                ha="center",
                va="center",
                color="white" if data[linha, coluna] > meio else "black",
            )

    save_or_show(save_path)


if __name__ == "__main__":
    perdas_treino = [12.0, 8.5, 6.1, 4.9, 4.2, 3.8, 3.5, 3.3, 3.2, 3.1]
    perdas_teste = [13.1, 9.4, 7.0, 5.8, 5.1, 4.7, 4.5, 4.4, 4.4, 4.3]

    plot_losses(
        perdas_treino,
        perdas_teste,
        save_path=REPORTS_DIR / "curva_de_perda.png",
    )
    plot_confusion_matrix(
        [[820, 40], [55, 185]],
        save_path=REPORTS_DIR / "matriz_confusao.png",
    )
