import torch
from tqdm import tqdm
from triplet_pytorch import *

LEARNING_RATE = 0.005
DEVICE = get_default_device()

model = ResNet_Triplet()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
criterion = TripletLoss()

checkpoint = torch.load("trained_model.pth")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


train_results = []
labels = []

model.eval()
with torch.no_grad():
    for img, _, _, label in tqdm(train_dl):
        train_results.append(model(img.to(device)).cpu().numpy())
        labels.append(label)
        
train_results = np.concatenate(train_results)
labels = np.concatenate(labels)
print(train_results.shape)
