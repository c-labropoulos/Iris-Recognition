import os
import torch
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision import models
from sklearn.decomposition import PCA
from sklearn import svm, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

TRAIN_PERC = 0.75


def main(path):
    # input transformations
    transimage = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    # data loader
    data = ImageFolder(root=path, transform=transimage)

    # split data into train and test datasets
    samples = len(data.targets)
    train_len = int(TRAIN_PERC * samples )
    test_len = samples - train_len
    train_dataset, test_dataset = random_split(data, (train_len, test_len))

    # dataset loaders
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)


    print("--- Original VGG-16:")
    vgg = models.vgg16(pretrained=True)
    print(vgg)


    vgg.eval()
    train_features = list()
    train_labels = list()
    test_features = list()
    test_labels = list()

    with torch.no_grad():
        for img, label in train_loader:
            train_features.append(vgg(img).squeeze().cpu().detach().numpy())
            train_labels.append(label.squeeze().cpu().detach().numpy())

        for img, label in test_loader:
            test_features.append(vgg(img).squeeze().cpu().detach().numpy())
            test_labels.append(label.squeeze().cpu().detach().numpy())

    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)


    pca = PCA(components=30)

    pca_train_feature = []

    for X in train_features:
        pca_train_feature.append(pca.fit_transform(X))

    pca_test_feature = []

    for X in test_features:
        pca_test_feature.append(pca.transform(X))



    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(pca_train_feature, train_labels)

    Y_predict = []
    for i in range(len(test_labels)):
        Y_predict.append(clf.predict(pca_test_feature[i]))

    print(metrics.classification_report(test_labels, Y_predict))


if __name__ == "__main__":
    path_mmu = "./mmu_iris"
    main(path_mmu)