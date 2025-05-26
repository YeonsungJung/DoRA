import os
import json
import torch
import glob
from PIL import Image
import torchvision.datasets as datasets

from .imagenet import project_logits
from .imagenet_classnames import get_classnames
import numpy as np


CLASS_SUBLIST = [
    409, 412, 414, 418, 419, 423, 434, 440, 444, 446, 455, 457, 462, 463, 470, 473, 479, 487, 499, 
    504, 507, 508, 518, 530, 531, 533, 539, 543, 545, 549, 550, 559, 560, 563, 567, 578, 587, 588, 589, 
    601, 606, 608, 610, 618, 619, 620, 623, 626, 629, 630, 632, 644, 647, 651, 655, 658, 659, 664, 671, 673, 677, 679, 689, 694, 695, 696, 
    700, 703, 720, 721, 725, 728, 729, 731, 732, 737, 740, 742, 749, 752, 759, 761, 765, 769, 770, 772, 773, 774, 778, 783, 790, 792, 797, 
    804, 806, 809, 811, 813, 828, 834, 837, 841, 842, 846, 849, 850, 851, 859, 868, 879, 882, 883, 893, 898, 
    902, 906, 907, 909, 923, 930, 950, 951, 954, 968, 999]
CLASS_SUBLIST_MASK = [(i in CLASS_SUBLIST) for i in range(1000)]



class ObjectNeObjectNetDatasetWithPaths():
    def __init__(self, path, transform, img_format="png"):
        self.loader = self.pil_loader
        self.img_format = img_format
        self.transform = transform
        with open(f"{path}/assets/folder_map.json", "r") as f:
            self.folder_map = json.load(f)
        
        files = glob.glob(path+"/images/**/*."+img_format, recursive=True)
        self.pathDict = {}
        for f in files:
            if f.split("/")[-2] not in self.folder_map: continue
            self.pathDict[f.split("/")[-1]] = f
        self.imgs = list(self.pathDict.keys())
        self.labels = [self.pathDict[img].split("/")[-2] for img in self.imgs]
        # self.labels = [self.folder_map[self.pathDict[img].split("/")[-2]] for img in self.imgs]

    def __getitem__(self, index):
        img = self.getImage(index)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]

        return {
            'images': img,
            'labels': label,
        }

    def getImage(self, index):
        img = self.loader(self.pathDict[self.imgs[index]])

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)
        return img

    def __len__(self):
        return len(self.imgs)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class ObjectNet:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser('./data'),
        batch_size=32,
        num_workers=32,
        classnames='objectnet_classnames',
        custom=False,
        method=None,
        flag=None,
    ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.custom = custom
        self.method = method
        self.flag = flag
        self.populate_train()
        self.populate_test()
    
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        # self.classnames = [self.classnames[i] for i in class_sublist]
        

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler())

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ObjectNeObjectNetDatasetWithPaths(self.location,
                                    transform=self.preprocess)

    def name(self):
        return 'objectnet'

    def get_class_sublist_and_mask(self):
        return CLASS_SUBLIST, CLASS_SUBLIST_MASK

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)