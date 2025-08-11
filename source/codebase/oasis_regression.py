import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
from PIL import Image
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"]=""

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
    response_data = pd.read_csv(response_filepath)
    response_data['image_name'] = response_data['theme'] + '.jpg'
    response_data['thing'] = response_data['theme'].str.replace('\d+', '',  regex=True)
    return response_data

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
    
    print('Now extracting features from {} on {}...'.format(model_string, 'CPU' if not torch.cuda.is_available() else 'GPU'))
    
    model_options = get_model_options()
    image_transforms = get_recommended_transforms(model_string)
        
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir,'{}.csv'.format(model_string))
    if not os.path.exists(output_file):
        
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

        stimulus_loader = DataLoader(dataset=StimulusSet(image_df, root, image_transforms), batch_size=128)
        stimulus_features = get_all_feature_maps(model, stimulus_loader, numpy=False)
        
        response_data = process_response_data('response/oasis_means_per_image.csv')
        response_data = deepcopy(image_df).merge(response_data, on = 'image_name')
        
        reg_dictlist = []
        for model_layer_index, model_layer in enumerate(tqdm(stimulus_features)):
            target_features = stimulus_features[model_layer].numpy()

            for category in response_data['category'].unique().tolist() + ['Combo']:
                if category != 'Combo':
                    response_data_sub = response_data.query("category == '{}'".format(category))
                    image_indices = response_data_sub.index.to_numpy()
                if category == 'Combo':
                    response_data_sub = response_data
                    image_indices = response_data_sub.index.to_numpy()

                for measurement in ['arousal','valence','beauty']:

                    y = response_data_sub[measurement].to_numpy()
                    X = target_features[image_indices,:]
                    regression = RidgeCV(alphas=np.logspace(-1,10,100), store_cv_values=True, 
                                         scoring='explained_variance').fit(X,y)

                    reg_dictlist.append({'model': model_name, 'train_type': train_type, 
                                         'model_layer': model_layer, 
                                         'model_layer_depth': model_layer_index+1, 
                                         'category': category, 'measurement': measurement,
                                         'score': regression.best_score_, 'penalty': regression.alpha_})

        reg_results = pd.DataFrame(reg_dictlist)
        reg_results.to_csv(output_file, index = None)

