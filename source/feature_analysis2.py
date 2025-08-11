from warnings import filterwarnings
filterwarnings("ignore")

import os, sys, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
from PIL import Image
from copy import copy

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('../model_opts')
from feature_extraction import *
from feature_reduction import *
from model_options import *

from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

from sklearn.metrics import explained_variance_score as ev_score
from sklearn.preprocessing import scale

def pearson_r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

scoring_metrics = {'ev_score': ev_score, 
                   'pearson_r': pearson_r_score}

from processing import *

def treves_rolls(x):
    if isinstance(x, np.ndarray):
        return ((np.sum(x / x.shape[0]))**2 / np.sum(x**2 / x.shape[0]))
    if isinstance(x, torch.Tensor):
        return ((torch.sum(x / x.shape[0]))**2 / torch.sum(x**2 / x.shape[0]))
    
#source: https://tntorch.readthedocs.io/en/latest/_modules/metrics.html

def torch_skewness(x):
    return torch.mean(((x - torch.mean(x))/torch.std(x))**3)

def torch_kurtosis(x, fisher=True):
    return torch.mean(((x-torch.mean(x))/torch.std(x))**4) - fisher*3

def torch_frobnorm(x):
    return torch.sqrt(torch.clamp(torch.dot(x,x), min=0))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Feature Extraction for Aesthetics Dataset')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--imageset', type=str, required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', type=str, required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--cuda_device', required=False, default='6',
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_string = args.model_string
    imageset = args.imageset
    output_type = args.output_type
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now processing {} with {} on {}...'.format(model_string, device_name, imageset))
    
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']

    output_files = {}
    for analysis in ['reg_redux']:
        output_dir = os.path.join('incoming', analysis, imageset)
        file_name = '{}.csv'.format(model_string.replace('/','-'))
        output_file = os.path.join(output_dir, file_name)
        print('Saving {} results to {}...'.format(analysis, output_file))
        output_files[analysis] = output_file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        
        root = 'images/{}/'.format(imageset)
        assets = glob(root + '*.jpg')
        asset_dictlist = []
        for asset in assets:
            imgstr = asset.split('/')[-1]
            row = {'image_name': imgstr,
                   'image_path': asset}
            asset_dictlist.append(row)
        image_df = (pd.DataFrame(asset_dictlist)
                    .sort_values(by='image_name', ignore_index=True))
        
        def image_transforms(image):
            return get_recommended_transforms(model_string)(transforms.Resize((256,256))(image))

        stimulus_loader = DataLoader(dataset=StimulusSet(image_df.image_path, image_transforms), batch_size=64)
        stimulus_features = srp_extraction(model_string, inputs = stimulus_loader,
                                           output_dir = 'srp_arrays/{}'.format(imageset))
        
        response_data = load_response_data(imageset, 'response')
        response_data = copy(image_df).merge(response_data, on = 'image_name')
        
        if not os.path.exists(output_files['reg_redux']):
        
            reg_dictlist = []
            for model_layer_index, model_layer in enumerate(tqdm(stimulus_features, desc = 'Regression (Layer)')):
                target_features = stimulus_features[model_layer]
                if isinstance(stimulus_features[model_layer], torch.Tensor):
                    target_features = target_features.numpy()

                measurements = [column for column in response_data.columns if column in ['arousal','beauty','valence']]
                for measurement in measurements:
                    for image_type in response_data['image_type'].unique().tolist() + ['Combo']:
                        if image_type != 'Combo':
                            response_data_sub = response_data[response_data['image_type'] == image_type]
                        if image_type == 'Combo':
                            response_data_sub = response_data

                        image_indices = response_data_sub.index.to_numpy()

                        y = response_data_sub[measurement].to_numpy()
                        X = scale(target_features[image_indices,:])
                        #alpha_values = np.logspace(-1,5,25).tolist()
                        alpha_values = [1000]
                        regression = RidgeCV(alphas=alpha_values, store_cv_values=True,
                                             scoring='explained_variance').fit(X,y)
                        
                        ridge_gcv_score, ridge_gcv_alpha = regression.best_score_, regression.alpha_
                        y_pred = regression.cv_values_[:, alpha_values.index(ridge_gcv_alpha)]
                        
                        for alpha_value in alpha_values:
                            y_pred = regression.cv_values_[:, alpha_values.index(alpha_value)]
                            
                            for score_type in scoring_metrics:
                                ridge_gcv_score = scoring_metrics[score_type](y, y_pred)

                                reg_dictlist.append({'model': model_name, 'train_type': train_type, 
                                                     'model_layer_index': model_layer_index+1,
                                                     'model_layer': model_layer,
                                                     'measurement': measurement,
                                                     'image_type': image_type,
                                                     'score_type': score_type,
                                                     'score': ridge_gcv_score, 
                                                     #'penalty': regression.alpha_,
                                                     'alpha': alpha_value})

            reg_results = pd.DataFrame(reg_dictlist)
            reg_results.to_csv(output_files['reg_redux'], index = None)
            #reg_results.to_parquet(output_files['reg_redux'].replace('.csv','.parquet'), index = None)

