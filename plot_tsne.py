from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(x, y, exp_dir, file_name):
    tsne = TSNE(n_components=2, init='pca', random_state=501)

    x_tsne = tsne.fit_transform(x)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, #cmap=ListedColormap(["black", "blue", "green", "red"]),
                linewidths=5, alpha=0.7, s=2)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.show()
    plt.savefig(exp_dir+'/tsne/'+file_name)


def plot_according_to_data():
    path = 'alpha100_test_index1/plot_data/'
    x = np.loadtxt(path+'feature_epoch0.txt')
    y = np.loadtxt(path+'pred_cluster4_epoch0.txt')
    plot_tsne(x,y,'alpha100_test_index1','new_tsne2.png')

if __name__ == '__main__':
    plot_according_to_data()

    pass