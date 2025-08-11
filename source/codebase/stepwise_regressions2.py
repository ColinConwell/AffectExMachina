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

sys.path.append('../mouseland/model_opts')
from model_options import *

def process_response_data(response_filepath):
    all_response_data = (pd.read_csv(response_filepath)
                 .groupby(['Subj','ImageType','Image'])
                 .agg({'Rating': 'mean', 'RT': 'mean'}).reset_index())
    all_response_data.columns = ['subject','image_type','image_name','rating','reaction_time']
    return all_response_data.groupby(['image_type','image_name'])['rating'].mean().reset_index()

def process_model_data(model_string, orient='wide'):
    model_data = (pd.read_csv('image_metrics/{}.csv'.format(model_string)))
    model_data = model_data.drop(['model_layer_index'], axis = 1)
    model_data = model_data.rename(columns={'image': 'image_name'})
    data_wide = pd.merge(model_data, response_data, on = 'image_name')
    data_wide['model_layer_depth'] = data_wide['model_layer_depth'] + 1
    id_columns = ['image_name','image_type','model','train_type','model_layer','model_layer_depth','rating']
    data_wide = data_wide[id_columns + [col for col in data_wide.columns.to_list() if col not in id_columns]]
    data_long = pd.melt(data_wide, id_vars=id_columns, 
                var_name = 'metric', value_name='value')
    
    if orient == 'wide':
        return(data_wide)
    if orient == 'long':
        return(data_long)
    
def process_corr_data(data_wide, orient='long'):
    model_layers = data_wide['model_layer'].unique().tolist()
    id_columns = ['model','train_type','image_type','model_layer', 'model_layer_depth']
    corr_data_wide = (data_wide.groupby(id_columns)
             .corrwith(data_wide['rating']).reset_index().drop('rating',axis = 1))
    corr_data_long = pd.melt(corr_data_wide, id_vars = id_columns, 
                             var_name = 'metric', value_name='corr')
    
    if orient == 'wide':
        return(corr_data_wide)
    if orient == 'long':
        return(corr_data_long)

if __name__ == "__main__":
    
    response_data = process_response_data('aesthetic_responses.csv')
    model_options = get_model_options()

    model_csvs = glob('image_metrics/*.csv')
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
            data_wide = process_model_data(target_model)
            model_layers = data_wide['model_layer'].unique()
            model_name = model_options[target_model]['model_name']
            train_type = model_options[target_model]['train_type']
            data_long = process_model_data(target_model, orient='long')
                
            #for metric in tqdm(data_long['metric'].unique(), leave=False):
            for metric in tqdm(['mean_activity','sparseness'], leave=False):
                temp_path = 'incoming/stepwise2/{}'.format(target_model)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                temp_file = '{}/{}.csv'.format(temp_path, metric)
                if os.path.exists(temp_file):
                    results_dflist.append(pd.read_csv(temp_file))
                 
                if not os.path.exists(temp_file):
                    results_dictlist = []
                    for image_type in tqdm(data_long['image_type'].unique(), leave=False):
                        running_model_layer_list = []
                        for model_layer_index, model_layer in enumerate(tqdm(model_layers, leave = False)):
                            running_model_layer_list.append(model_layer)

                            data_i = data_wide[(data_wide['image_type'] == image_type)]
                            y = data_i[(data_i['model_layer']==model_layers[0])]['rating'].to_numpy()
                            X = np.stack([data_i[(data_i['model_layer']==model_layer)][metric].to_numpy() 
                                          for model_layer in running_model_layer_list], axis = 1)
                            alpha_values = np.array([0.01, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0])
                            regression = RidgeCV(alphas=alpha_values, store_cv_values=True, 
                                                 scoring='explained_variance').fit(X,y)
                            ridge_gcv_score, ridge_gcv_alpha = regression.best_score_, regression.alpha_
                            results_dictlist.append({'model': model_name, 'train_type': train_type,
                                                     'image_type': image_type, 'metric': metric,
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
