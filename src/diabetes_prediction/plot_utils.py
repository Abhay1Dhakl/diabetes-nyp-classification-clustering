from __future__ import annotations

import matplotlib.pyplot as plt


def plot_confusion_matrix(labels, cm, output_path) -> None:
    """Plot and save a confusion matrix.

    Args:
        labels: Class labels for axes.
        cm: Confusion matrix array.
        output_path: File path for the saved image.
    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
