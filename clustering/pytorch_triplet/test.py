import torch
from tqdm import tqdm
from triplet_pytorch import *

from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA


# data
data_dir = "../../rawdata/logos"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
# clustering
pca_dim = 50
n_clusters = 64

IMAGE_SIZE = 256
LEARNING_RATE = 0.005
DEVICE = get_default_device()

model = ResNet_Triplet()

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
criterion = TripletLoss()
checkpoint = torch.load("trained_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

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
            feature = model.Feature_Extractor(tensor.cuda())
        features.append(feature.reshape(-1).cpu().numpy())

pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
reduced = pca.fit_transform(features)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
pseudo_labels = list(kmeans.fit_predict(reduced))


result = {
    "Filenames": file_paths,
    "Labels": pseudo_labels
}

pickle_path = '../../rawdata/tripletmining.pickle'
with open(pickle_path, 'wb') as f:
    pickle.dump(result, f)

def open_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


data = open_pickle(pickle_path)
print(data["Labels"][:3])
