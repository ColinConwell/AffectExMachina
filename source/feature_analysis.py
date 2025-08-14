from warnings import filterwarnings
filterwarnings("ignore")

import os, argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import copy

from torch.utils.data import DataLoader

# import model options
from .model_opts import *

from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

from sklearn.metrics import explained_variance_score as ev_score
from sklearn.preprocessing import scale

def pearson_r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

scoring_metrics = {'ev_score': ev_score, 
                   'pearson_r': pearson_r_score}

from dataset import *

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

def prepare_output_files(output_dir, output_type, analyses, imageset, model_string):
    if not isinstance(analyses, list):
        analyses = [analyses]
        
    output_files = {}
    for analysis in analyses:
        output_dir = os.path.join(output_dir, analysis, imageset)
        file_name = '{}.{}'.format(model_string.replace('/','-'), output_type)
        output_file = os.path.join(output_dir, file_name)
        print('Saving {} results to {}...'.format(analysis, output_file))
        output_files[analysis] = output_file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    return output_files

def get_stimulus_loader(image_paths, image_transforms, batch_size = 64):
    return DataLoader(dataset=StimulusSet(image_paths, image_transforms), batch_size=64)
    

def get_feature_metrics(model_option, features, image_df):
    model_name = model_option['model_name']
    train_type = model_option['train_type']
    
    metric_dictlist = []
    for model_layer_index, model_layer in enumerate(tqdm(features, desc = 'Metrics (Layer)')):
        target_map = features[model_layer]
        for target_i, target_activity in enumerate(target_map):
            image_name = image_df.image_name.iloc[target_i]

            mean_activity = target_activity.mean().item()
            mean_absolute = target_activity.abs().mean().item()
            max_activity = target_activity.max().item()
            min_activity = target_activity.min().item()
            var_activity = target_activity.std().item()
            var_absolute = target_activity.abs().std().item()
            sparseness = treves_rolls(target_activity).item()
            skewness = torch_skewness(target_activity.abs()).item()
            kurtosis = torch_kurtosis(target_activity.abs()).item()
            frobenius = torch_frobnorm(target_activity.abs()).item()
            activity_range = max_activity - min_activity

            metric_dictlist.append({
                'image': image_name, 
                'model': model_name,
                'train_type': train_type,
                'model_layer': model_layer, 
                'model_layer_index': model_layer_index + 1,
                'mean_absolute': mean_absolute,
                'mean_activity': mean_activity,
                'var_activity': var_activity,
                'var_absolute': var_absolute,
                'max_activity': max_activity,
                'min_activity': min_activity,
                'range': activity_range,
                'sparseness': sparseness,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'frobenius': frobenius,
            })

    return pd.DataFrame(metric_dictlist)

def get_regression_results(model_option, stimulus_features, response_data, 
                           alpha_values = np.logspace(-1,5,25).tolist()):
    
    model_name = model_option['model_name']
    train_type = model_option['train_type']
        
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
                regression = RidgeCV(alphas=alpha_values, store_cv_values=True,
                                     scoring='explained_variance').fit(X,y)

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
                                             'alpha': alpha_value})

    return pd.DataFrame(reg_dictlist)   

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Feature Extraction for Aesthetics Dataset')
    parser.add_argument('--model_string', type=str, required=True, 
                        help='name of deep net model to load')
    parser.add_argument('--imageset', required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--output_dir', required=False, default='incoming', 
                        help='destination for output files')
    parser.add_argument('--cuda_device', required=False, default='6',
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_string = args.model_string
    imageset = args.imageset
    output_type = args.output_type
    output_dir = args.output_dir
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now processing {} with {} on {}...'.format(model_string, device_name, imageset))
    
    model_option = get_model_options()[model_string]
    analyses = ['reg_redux']

    output_files = prepare_output_files(output_dir, output_type, analyses, imageset, model_string) 
    
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        
        image_data = load_image_data(imageset)
        image_transforms = get_recommended_transforms(model_string)

        stimulus_loader = get_stimulus_loader(image_data.image_path, image_transforms)
        stimulus_features = get_all_feature_maps(model_string, inputs = stimulus_loader)
        
        response_data = load_response_data(imageset, 'response')
        response_data = copy(image_data).merge(response_data, on = 'image_name')
        
        if 'metrics' in output_files and not os.path.exists(output_files['metrics']):
            
            metric_data = get_feature_metrics(model_option, stimulus_features)
            metric_data.to_csv(output_files['metrics'], index = None)
        
        if not os.path.exists(output_files['reg_redux']):
            
            stimulus_features = get_feature_map_srps(stimulus_features, delete_originals = True)
            
            reg_results = get_regression_results(model_option, stimulus_features, 
                                                 response_data, alpha_values = [1000])
            
            reg_results.to_csv(output_files['reg_redux'], index = None)

