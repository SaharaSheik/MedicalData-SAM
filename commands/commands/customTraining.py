#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from PIL import Image
from patchify import patchify
import random
from scipy import ndimage
from imageMaker import display_images_in_folder
from torch.utils.data import Dataset
from transformers import SamModel, SamConfig, SamProcessor
import torch
from transformers import SamProcessor
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
from torch.nn.functional import threshold, normalize
from transformers import SamModel
from datasets import Dataset
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

## creating image paths

imagepath = '/home/cap5415.student17/VisionProject/SAM2/segment-anything/data/2d_images/'
maskpath = '/home/cap5415.student17/VisionProject/SAM2/segment-anything/data/2d_masks/'
large_images = tifffile.imread("/home/cap5415.student17/VisionProject/SAM2/segment-anything/data/2d_images/ID_0000_Z_0142.tif")
large_masks = tifffile.imread("data/2d_masks/ID_0000_Z_0142.tif")


print(large_images.shape[0])

## crearting a list of image files
imgageFiles = display_images_in_folder(imagepath)
print(imgageFiles[20])

# Patch sieze and step size so they dont overlap # tranformer base
patch_size = 256
step = 256

all_img_patches = []
for img in range(len(imgageFiles)):
    large_image = tifffile.imread(imagepath+imgageFiles[img])
    patches_img = patchify(large_image, (patch_size, patch_size), step=step)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)


MaskFiles = display_images_in_folder(maskpath)
print(MaskFiles[20])

all_mask_patches = []
for img in range(len(imgageFiles)):
    large_mask = tifffile.imread(maskpath+MaskFiles[img])
    patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i,j,:,:]
            single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)


print(images.shape)



# find empty masks (the ones fully black or fully white as they provide challenges to during training)
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
print("Image shape:", filtered_images.shape)
print("Mask shape:", filtered_masks.shape)

#getting data readty for training pipeline in form of a dict

dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray(mask) for mask in filtered_masks],
}

# Create the dataset
dataset = Dataset.from_dict(dataset_dict)

print(dataset)

## Display a few images and their masks for visualization

img_num = random.randint(0, filtered_images.shape[0]-1)
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))


axes[0].imshow(np.array(example_image), cmap='gray')
axes[0].set_title("Image")


axes[1].imshow(example_mask, cmap='gray')


for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


plt.show()

#create bounding box function

def get_bounding_box(ground_truth_map):

  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)

  # add coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


## create a pipline for Sam Dataset

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box
    prompt = get_bounding_box(ground_truth_mask)


    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # process batch dimention so model does no see this
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


## use hugging face trandormers to import SAM

from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)

# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)


print(batch["ground_truth_mask"].shape)

# Load  model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# We are only training the mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)



from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#differnce factors can be used we use Dice
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')



#Training
num_epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    save_dir = '/home/cap5415.student17/VisionProject/SAM2/segment-anything/checkpoints'
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')


    save_path = os.path.join(save_dir, f'model_weights_epoch{epoch}.pth')
    torch.save(model.state_dict(), save_path)

    print(f'Model weights saved at epoch {epoch} in {save_path}')

    print(f'Model saved at epoch {epoch}')

    #torch.save(model.state_dict(), "/home/cap5415.student17/VisionProject/SAM2/segment-anything/checkpoints/model_checkpoint"+str(epoch)+".pth")


# Save the model's state dictionary to a file
#torch.save(model.state_dict(), "/home/cap5415.student17/VisionProject/SAM2/segment-anything/checkpoints/model_checkpoint.pth")


# Load the model configuration can use vit base, vit huge or vit large
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model
model = SamModel(config=model_config)
#load preloaded model
model.load_state_dict(torch.load("checkpoints/model_weights_epoch49.pth"))


# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# let's take a random training example
idx = random.randint(0, filtered_images.shape[0]-1)

# load image
test_image = dataset[idx]["image"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()

# forward pass
with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

# apply sigmoid
seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
seg_prob = seg_prob.cpu().numpy().squeeze()
medsam_seg = (seg_prob > 0.5).astype(np.uint8)



## Plotting Images


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.array(test_image), cmap='gray')

axes[1].imshow(medsam_seg, cmap='gray')
axes[1].set_title("Mask")

axes[2].imshow(seg_prob)
axes[2].set_title("Probability Map")


for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()