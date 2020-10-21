import numpy as np
import matplotlib.pyplot as plt


def plot_embedding(z, labels, title_str=None, save_str=None):
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    if title_str:
        plt.title(title_str, fontsize=24)
    if save_str:
        plt.savefig(save_str + '.png')
    plt.close()


def plot_reconstructed(x, labels, n=10, save_str=None):
    num_classes = len(np.unique(labels))
    img_idx = [np.where(labels == i)[0][j] for i in range(num_classes) for j in range(n)]

    merged_img = np.zeros((28 * num_classes, 28 * n))
    for idx, image in enumerate(x[img_idx]):
        j = int(idx % n)  # j-th sample of i-th class
        i = int(idx / n)
        merged_img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.squeeze(image, axis=2)
    ax = plt.axes()
    plt.imshow(merged_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if save_str:
        plt.savefig(save_str + '.png')
    plt.close()
