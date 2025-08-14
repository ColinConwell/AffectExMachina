import os, sys, json
import numpy as np
import pandas as pd
import seaborn as sns

from copy import copy
from glob import glob
from tqdm.auto import tqdm as tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import explained_variance_score as ev_score
from sklearn.linear_model import LinearRegression, RidgeCV
from scipy.stats import pearsonr

sys.path.append('model_opts')
from model_options import *

def load_response_data(imageset, response_dir = 'response'):
    response_data = pd.read_csv(os.path.join(response_dir, '{}_means_per_image.csv').format(imageset))
    if imageset == 'oasis':
        response_data = response_data.rename(columns={'category': 'image_type'})
        response_data['image_name'] = response_data['theme'] + '.jpg'
        response_data = response_data.drop(['item','theme'], axis = 1)
    if imageset == 'vessel':
        response_data = response_data.rename(columns={'rating': 'beauty'})
    return response_data

response_data = {'vessel': load_response_data('vessel'), 'oasis': load_response_data('oasis')}

def process_metric_data(model_string, dataset, orient='wide'):
    model_data = (pd.read_csv('metrics/{}/{}.csv'.format(dataset,model_string)))
    if 'model_layer_index' in model_data.columns:
        model_data = model_data.drop(['model_layer_index'], axis = 1)
    if 'image' in model_data.columns:
        model_data = model_data.rename(columns={'image': 'image_name'})
    
    data_wide = pd.merge(model_data, response_data[dataset], on = 'image_name')
    data_wide['model_layer_index'] = data_wide['model_layer_depth'] + 1
    data_wide['model_layer_depth'] = (data_wide['model_layer_index'] / 
                                      data_wide['model_layer'].nunique())
    
    id_columns = ['image_name','image_type','model','train_type',
                  'model_layer','model_layer_index','model_layer_depth']
    measurement_columns = [col for col in data_wide.columns 
                           if col in ['arousal','beauty','valence']]
    
    analysis_columns = [col for col in data_wide.columns 
                        if col not in id_columns + measurement_columns]
    
    data_wide = data_wide[id_columns + measurement_columns + analysis_columns]
    data_wide = pd.melt(data_wide, id_vars=id_columns + analysis_columns, 
                        var_name = 'measurement', value_name='rating')
    
    data_long = pd.melt(data_wide, id_vars=id_columns + ['measurement', 'rating'], 
                        var_name = 'metric', value_name='value')
    
    if orient == 'wide':
        return(data_wide)
    if orient == 'long':
        return(data_long)
    
def process_corr_data(data_wide, orient='long'):
    model_layers = data_wide['model_layer'].unique().tolist()
    id_columns = ['model','train_type','image_type','model_layer',
                  'model_layer_index','model_layer_depth', 'measurement']
    
    corr_data_wide = (data_wide.groupby(id_columns)
             .corrwith(data_wide['rating']).reset_index().drop('rating',axis = 1))
    corr_data_long = pd.melt(corr_data_wide, id_vars = id_columns, 
                             var_name = 'metric', value_name='corr')
    
    if orient == 'wide':
        return(corr_data_wide)
    if orient == 'long':
        return(corr_data_long)

if __name__ == "__main__":
    
    model_options = get_model_options()

    model_csvs = glob('metrics/oasis/*.csv')
    target_models = [csv.split('/')[1].split('.')[0] for csv in model_csvs]
    #target_models = ['alexnet','vgg11','vgg13','vgg16','vgg19','resnet18','resnet34']

    output_file = 'stepwise_regressions2.csv'
    if os.path.exists(output_file):
        reg_results = pd.read_csv(output_file)

    if not os.path.exists(output_file):
        results_dflist = []
        iterator = tqdm(target_models)
        for target_model in iterator:
            iterator.set_description(target_model) 
            for dataset in tqdm(['vessel','oasis'], leave = False):
                data_wide = process_metric_data(target_model, dataset)
                model_layers = data_wide['model_layer'].unique()
                model_name = model_options[target_model]['model_name']
                train_type = model_options[target_model]['train_type']

                for metric in tqdm(['mean_activity','sparseness'], leave=False):
                    temp_path = 'incoming/stepwise3/{}'.format(target_model)
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                    temp_file = '{}/{}.csv'.format(temp_path, metric)
                    if os.path.exists(temp_file):
                        results_dflist.append(pd.read_csv(temp_file))

                    if not os.path.exists(temp_file):
                        results_dictlist = []
                        for measurement in tqdm(data_wide['measurement'].unique(), leave = False):
                            for image_type in tqdm(data_wide['image_type'].unique(), leave=False):
                                running_model_layer_list = []
                                for model_layer_index, model_layer in enumerate(tqdm(model_layers, leave = False)):
                                    running_model_layer_list.append(model_layer)
                                    data_i = data_wide[(data_wide['image_type'] == image_type) & 
                                                   (data_wide['measurement'] == measurement)]
                                    y = data_i[(data_i['model_layer']==model_layers[0])]['rating'].to_numpy()
                                    X = np.stack([data_i[(data_i['model_layer']==model_layer)][metric].to_numpy() 
                                                  for model_layer in running_model_layer_list], axis = 1)
                                    alpha_values = [0.01, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 100.0]
                                    regression = RidgeCV(alphas=alpha_values, store_cv_values=True, 
                                                         scoring='explained_variance').fit(X,y)
                                    ridge_gcv_score, ridge_gcv_alpha = regression.best_score_, regression.alpha_
                                    ridge_cv_values = regression.cv_values_[:, alpha_values.index(ridge_gcv_alpha)]

                                    for score_type in scoring_metrics:
                                        ridge_gcv_score = scoring_metrics[score_type](y, ridge_cv_values)
                                    
                                        results_dictlist.append({'model': model_name, 'train_type': train_type,
                                                                 'image_type': image_type, 'metric': metric,
                                                                 'measurement': measurement,
                                                                 'score_type': score_type,
                                                                 'score': ridge_gcv_score, 
                                                                 'alpha': ridge_gcv_alpha,
                                                                 'model_layer': model_layer,
                                                                 'model_depth': len(model_layers),
                                                                 'model_layer_index': model_layer_index + 1,
                                                                 'model_layer_depth': (model_layer_index + 1) / len(model_layers)})

                        incoming_results = pd.DataFrame(results_dictlist)
                        incoming_results.to_csv(temp_file, index = None)
                        results_dflist.append(incoming_results)

        results = pd.concat(results_dflist)
        results.to_csv(output_file, index = None)
