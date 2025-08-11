import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdm
from glob import glob
from PIL import Image

import torch, torchvision
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('../mouseland/model_opts')
from feature_extraction import *
from model_options import *
from image_ops import *

torch.cuda.set_device(3)

class StimulusSet(Dataset):
    def __init__(self, csv, root_dir, image_transforms=None):
        
        self.root = os.path.expanduser(root_dir)
        self.transforms = image_transforms
        
        if isinstance(csv, pd.DataFrame):
            self.df = csv
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        
        self.images = self.df.ImageName

    def __getitem__(self, index):
        filename = os.path.join(self.root, self.images.iloc[index])
        img = Image.open(filename).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
            
        return img
    
    def __len__(self):
        return len(self.images)
    
def torch_cosine(x1, x2=None, eps=1e-6, distance=False):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        cos = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        if not distance:
            return cos 
        if distance:
            return 1 - cos
        
def treves_rolls(x):
    if isinstance(x, np.ndarray):
        return ((np.sum(x / x.shape[0]))**2 / np.sum(x**2 / x.shape[0]))
    if isinstance(x, torch.Tensor):
        return ((torch.sum(x / x.shape[0]))**2 / torch.sum(x**2 / x.shape[0]))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature Extraction for Aesthetics Dataset')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--output_dir', type=str, required=False, default='features',
                        help='output directory for results')
    parser.add_argument('--cuda_device', type=int, required=False, default=0,
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    model_string = args.model_string
    
    print('Now extracting featues from {}...'.format(model_string))
    
    model_options = get_model_options(train_type='imagenet')
    image_transforms = get_image_transforms()['imagenet']
        
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir,'{}.csv'.format(model_name))
    if not os.path.exists(output_file):
        
        root = 'vessel_assets/'
        assets = glob(root + '*.jpg')
        dictlist = []
        for asset in assets:
            imgstr = asset.split('/')[1]
            row = {'ImageName': imgstr}
            dictlist.append(row)
        image_df = pd.DataFrame(dictlist)
        
        model = eval(model_call)
        model = model.eval()
        model = model.cuda()

        imagenet_images = np.load('imagenet_sample.npy')

        stimulus_loader = DataLoader(dataset=StimulusSet(image_df, root, image_transforms), batch_size=64)
        random_loader = DataLoader(dataset=Array2DataLoader('random_noise.npy', image_transforms), batch_size=64)
        imagenet_loader = DataLoader(dataset=Array2DataLoader(imagenet_images[:1000], image_transforms), batch_size=64)

        stimulus_features = get_all_feature_maps(model, stimulus_loader, numpy=False)
        random_features = get_all_feature_maps(model, random_loader, numpy=False)
        imagenet_features = get_all_feature_maps(model, imagenet_loader, numpy=False)

        
        activity_dictlist = []
        for map_key in tqdm(list(stimulus_features.keys())):
            target_map = stimulus_features[map_key]
            random_map = random_features[map_key]
            imagenet_map = imagenet_features[map_key]

            target_activity_sum = target_map.abs().sum(axis=1)
            target_activity_mean = target_map.abs().mean(axis=1)
            random_cosines = torch_cosine(target_map, random_map).mean(axis=1)
            random_euclids = torch.cdist(target_map, random_map).mean(axis=1)
            imagenet_cosines = torch_cosine(target_map, imagenet_map).mean(axis=1)
            imagenet_euclids = torch.cdist(target_map, imagenet_map).mean(axis=1)
            mean_random_cosine = torch_cosine(target_map, random_map.mean(axis=0).unsqueeze(0)).squeeze()
            mean_random_euclid = torch.cdist(target_map, random_map.mean(axis=0).unsqueeze(0)).squeeze()
            mean_imagenet_cosine = torch_cosine(target_map, imagenet_map.mean(axis=0).unsqueeze(0)).squeeze()
            mean_imagenet_euclid = torch.cdist(target_map, imagenet_map.mean(axis=0).unsqueeze(0)).squeeze()

            for target_i, target_activity in tqdm(enumerate(target_map), leave=False):
                image_name = image_df.ImageName.iloc[target_i]
                
                activity_dictlist.append({
                    'image_name': image_name,
                    'model': model_name, 
                    'model_layer': map_key, 
                    'model_layer_index': map_key.split('-')[1],
                    'sparseness': treves_rolls(target_activity).item(),
                    'mean_activity': target_activity_mean[target_i].item(),
                    'cosine_to_noise_mean': mean_random_cosine[target_i].item(),
                    'distance_to_noise_mean': mean_random_euclid[target_i].item(),
                    'cosine_to_imagenet_mean': mean_imagenet_cosine[target_i].item(),
                    'distance_to_imagenet_mean': mean_imagenet_euclid[target_i].item(),
                    'mean_cosine_to_noise': random_cosines[target_i].item(),
                    'mean_distance_to_noise': random_euclids[target_i].item(),
                    'mean_cosine_to_imagenet': imagenet_cosines[target_i].item(),
                    'mean_distance_to_imagenet': imagenet_euclids[target_i].item(),
                })

        activity_df = pd.DataFrame(activity_dictlist)
        activity_df.to_csv(output_file, index=None)

