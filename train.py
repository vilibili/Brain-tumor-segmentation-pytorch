import tqdm
import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.my_net import Mini_unet
from torch.utils.data import DataLoader
from utilities.dataReader import datareader
import SimpleITK as sitk

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# Hyper-parameters
input_size = 3
num_classes = 4
num_epochs = 25
batch_size = 2
learning_rate = 0.0001

image_test_path = r'dataset/.............'
mask_test_path = r'dataset\..............'
ckpt_path = r'ckpt\MiniUnet_model.pytorch'

img_t1ce_test = sitk.ReadImage(image_test_path)
img_t1ce_test = sitk.GetArrayFromImage(img_t1ce_test)[100]
img_t1ce_test = np.expand_dims(img_t1ce_test,axis=0)
img_t1ce_test = np.expand_dims(img_t1ce_test,axis=0)
img_t1ce_test = torch.from_numpy(img_t1ce_test)

mask_t1ce_test = sitk.ReadImage(mask_test_path)
mask_t1ce_test = sitk.GetArrayFromImage(mask_t1ce_test)[100]



model = Mini_unet(num_classes=num_classes)

model.to(device)

print(model)
dtset = datareader()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    dt_loader_train = DataLoader(dtset, batch_size=batch_size, shuffle=True)
    for batch_index, batch in enumerate(dt_loader_train):

        image = batch[0]
        mask = batch[1]

        image = image.to(device)
        mask = mask.to(device=device, dtype=torch.long)
        y_pred = model(image)

        loss = criterion(y_pred, mask)
        print('Epoch ',epoch,'  iter ',batch_index*2,'  loss : ', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), ckpt_path)

    with torch.no_grad():
        img_t1ce_test = img_t1ce_test.to(device)

        # Generate prediction
        prediction = model(img_t1ce_test)

        prediction = np.squeeze(prediction, axis=0)

        prediction = prediction.cpu().numpy()

        predicted_class = np.argmax(prediction, axis=0)

        predicted_class = np.array(predicted_class, dtype=np.uint8)

        predicted_class[predicted_class==3]=4

        # Show result
        plt.imshow(predicted_class, cmap='gray')
        plt.show()

        # Show mask
        plt.imshow(mask_t1ce_test, cmap='gray')
        plt.show()
