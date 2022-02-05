import os

import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset


class datareader(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_path = r'dataset\RSNA_ASNR_MICCAI_BraTS2021_TrainingData'):
        'Initialization'
        self.image_path = image_path
        self.folders_name = os.listdir(self.image_path)
        self.images, self.labels = self.get_images()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'

        #x = np.moveaxis(self.images[index], -1, 0)
        x = self.images[index]
        y = self.labels[index]

        return x, y

  def get_images(self):
      images = []
      masks = []
      for fld_name in self.folders_name:
          #path_img_flair = os.path.join(self.image_path, fld_name, fld_name + '_flair.nii.gz')
          #path_img_t1 = os.path.join(self.image_path, fld_name, fld_name + '_t1.nii.gz')
          path_img_t1ce = os.path.join(self.image_path, fld_name, fld_name + '_t1ce.nii.gz')
          #path_img_t2 = os.path.join(self.image_path, fld_name, fld_name + '_t2.nii.gz')
          path_label = os.path.join(self.image_path, fld_name, fld_name + '_seg.nii.gz')

          #img_flair = sitk.ReadImage(path_img_flair)
          #img_flair = sitk.GetArrayFromImage(img_flair)

          #img_t1 = sitk.ReadImage(path_img_t1)
          #img_t1 = sitk.GetArrayFromImage(img_t1)

          img_t1ce = sitk.ReadImage(path_img_t1ce)
          img_t1ce = sitk.GetArrayFromImage(img_t1ce)

          #img_t2 = sitk.ReadImage(path_img_t2)
          #img_t2 = sitk.GetArrayFromImage(img_t2)

          label = sitk.ReadImage(path_label)
          label = sitk.GetArrayFromImage(label)
          label[label==4]=3

          for index in range(0,img_t1ce.shape[0]):
              #img_flair_ = img_flair[index]
              #img_flair_ = np.expand_dims(img_flair_,axis=0)

              #img_t1_ = img_t1[index]
              #img_t1_ = np.expand_dims(img_t1_, axis=0)

              img_t1ce_ = img_t1ce[index]
              img_t1ce_ = np.expand_dims(img_t1ce_, axis=0)

              #img_t2_ = img_t2[index]
              #img_t2_ = np.expand_dims(img_t2_, axis=0)

              label_ = label[index]
              #label_ = np.expand_dims(label_, axis=0)

              images.append(img_t1ce_)
              masks.append(label_)

      images = np.array(images, dtype=np.uint8)
      masks = np.array(masks, dtype=np.uint8)
      return images, masks
