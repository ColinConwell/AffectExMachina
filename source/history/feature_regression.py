import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
from PIL import Image
from copy import deepcopy

import torch, torchvision
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ..model_opts import *

from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

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

def process_response_data(response_filepath):
    all_response_data = (pd.read_csv(response_filepath)
                 .groupby(['Subj','ImageType','Image'])
                 .agg({'Rating': 'mean', 'RT': 'mean'}).reset_index())
    all_response_data.columns = ['subject','image_type','image_name','rating','reaction_time']
    return all_response_data.groupby(['image_type','image_name'])['rating'].mean().reset_index()

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
    cuda_device = args.cuda_device
    torch.cuda.set_device(5)
    
    print('Now extracting features from {}...'.format(model_string))
    
    model_options = get_model_options()
    image_transforms = get_recommended_transforms(model_string)
        
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir,'{}.csv'.format(model_string))
    if not os.path.exists(output_file):
        
        root = 'vessel_assets/'
        assets = glob(root + '*.jpg')
        dictlist = []
        for asset in assets:
            imgstr = asset.split('/')[1]
            row = {'image_name': imgstr}
            dictlist.append(row)
        image_df = pd.DataFrame(dictlist)
        
        model = eval(model_call)
        model = model.eval()
        model = model.cuda()

        stimulus_loader = DataLoader(dataset=StimulusSet(image_df, root, image_transforms), batch_size=64)
        stimulus_features = get_all_feature_maps(model, stimulus_loader, numpy=False)
        
        response_data = process_response_data('aesthetic_responses.csv')
        response_data = deepcopy(image_df).merge(response_data, on = 'image_name')
        
        reg_dictlist = []
        for model_layer_index, model_layer in enumerate(tqdm(stimulus_features)):
            target_features = stimulus_features[model_layer].numpy()

            for image_type in response_data['image_type'].unique():
                response_data_sub = response_data.query("image_type == '{}'".format(image_type))
                image_indices = response_data_sub.index.to_numpy()

                y = response_data_sub.rating.to_numpy()
                X = target_features[image_indices,:]
                regression = RidgeCV(alphas=np.logspace(-1,10,100), store_cv_values=True, 
                                 scoring='explained_variance').fit(X,y)

                reg_dictlist.append({'model': model_name, 'train_type': train_type, 'model_layer': model_layer, 
                                     'model_layer_depth': model_layer_index+1, 'image_type': image_type, 
                                     'score': regression.best_score_, 'penalty': regression.alpha_})

        reg_results = pd.DataFrame(reg_dictlist)
        reg_results.to_csv(output_file, index = None)

