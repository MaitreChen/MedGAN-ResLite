import os.path
import itertools

from matplotlib import pyplot as plt
import numpy as np


def plot_g_d_loss(G_loss_set, D_loss_set, save_dir):
    plt.figure(figsize=(10, 5))
    plt.title("Generator & Discriminator Loss During Training")
    plt.plot(G_loss_set, label="G")
    plt.plot(D_loss_set, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/training_loss.png")
    plt.show()


def plot_accuracy(epochs, train_acc_set, test_acc_set, val_acc_set, save_dir):
    plt.figure(figsize=(8, 6))
    x = range(1, epochs + 1)
    plt.plot(x, train_acc_set, 'r', label='train-acc')
    plt.plot(x, test_acc_set, 'g', label='test-acc')
    plt.plot(x, val_acc_set, 'b', label='val-acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    # save result
    plt.savefig(f"{save_dir}/acc.png")
    plt.show()


def plot_confusion_matrix(cm, save_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    classes = ('normal', 'pneumonia')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    plt.rc('font', family='Times New Roman')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20, rotation=90)

    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=28)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    plt.savefig(os.path.join(save_dir, 'cm.jpg'), bbox_inches='tight')
    plt.show()
