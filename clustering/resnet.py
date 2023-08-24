import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.optim import Adam

import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

# data
data_dir = "../rawdata/logos"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
# clustering
pca_dim = 50
n_clusters = 64
# convnet
batch_size = 64
num_classes = 100
num_epochs = 2


model = resnet50(pretrained=True)
model = model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])
model.cuda()



transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features = []
for path in tqdm(file_paths):
    with Image.open(path).convert("RGB") as img:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = model(tensor.cuda())
        features.append(feature.reshape(-1).cpu().numpy())

pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
reduced = pca.fit_transform(features)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
pseudo_labels = list(kmeans.fit_predict(reduced))


result = {
    "Filenames": file_paths,
    "Labels": pseudo_labels
}

pickle_path = '../rawdata/resnet.pickle'
with open(pickle_path, 'wb') as f:
    pickle.dump(result, f)

def open_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


data = open_pickle(pickle_path)
print(data["Labels"][:3])