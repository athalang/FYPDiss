from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from train import TransformerLitModel
from dataset import ModCountDataModule

if __name__ == '__main__':
    dm = ModCountDataModule()
    dm.setup()

    model = TransformerLitModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
    model.eval()

    acts, labels = model.extract_activations(dm.test_dataloader())
    pca = PCA(n_components=2).fit_transform(acts.detach())
    plt.scatter(pca[:,0], pca[:,1], c=labels, alpha=0.5)
    plt.show()