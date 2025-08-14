import os, argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ..model_opts import *

class StimulusSet(Dataset):
    def __init__(self, csv, root_dir, image_transforms=None):
        
        self.root = os.path.expanduser(root_dir)
        self.transforms = image_transforms
        
        if isinstance(csv, pd.DataFrame):
            self.df = csv
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        
        self.images = self.df.image_name

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
    parser.add_argument('--imageset', type=str, required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_dir', type=str, required=False, default='metrics',
                        help='output directory for results')
    parser.add_argument('--cuda_device', type=int, required=False, default=0,
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    imageset = args.imageset
    model_string = args.model_string
    cuda_device = args.cuda_device
    output_dir = args.output_dir
    
    cuda_device = '' if cuda_device is None else str(cuda_device)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_device)
    
    print('Now extracting features from {}... on {}'.format(model_string))
    
    model_options = get_model_options()
    image_transforms = get_recommended_transforms(model_string)
        
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']

    output_dir = os.path.join(output_dir, imageset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir,'{}.csv'.format(model_string))
    if not os.path.exists(output_file):
        
        if imageset == 'vessel':
            root = 'vessel_assets/'
            assets = glob(root + '*.jpg')
            dictlist = []
            for asset in assets:
                imgstr = asset.split('/')[1]
                row = {'image_name': imgstr}
                dictlist.append(row)
            image_df = pd.DataFrame(dictlist)
            
        if imageset == 'oasis':
            root = 'images/oasis/'
            assets = glob(root + '*.jpg')
            dictlist = []
            for asset in assets:
                imgstr = asset.split('/')[-1]
                row = {'image_name': imgstr}
                dictlist.append(row)
            image_df = pd.DataFrame(dictlist).sort_values(by='image_name', ignore_index=True)
        
        model = eval(model_call)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        stimulus_loader = DataLoader(dataset=StimulusSet(image_df, root, image_transforms), batch_size=64)
        stimulus_features = get_all_feature_maps(model, stimulus_loader, numpy=False)
        
        activity_dictlist = []
        for map_key_i, map_key in enumerate(tqdm(stimulus_features)):
            target_map = stimulus_features[map_key]
            for target_i, target_activity in enumerate(target_map):
                image_name = image_df.image_name.iloc[target_i]

                activity_dictlist.append({
                    'image': image_name, 
                    'model': model_name, 
                    'train_type': train_type,
                    'model_layer': map_key, 
                    'model_layer_depth': map_key_i,
                    'mean_activity': target_activity.abs().mean().item(),
                    'sparseness': treves_rolls(target_activity).item()
                })

        activity_df = pd.DataFrame(activity_dictlist)
        activity_df.to_csv(output_file, index=None)

