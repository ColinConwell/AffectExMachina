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
 
def process_response_data(response_filepath):
    all_response_data = (pd.read_csv(response_filepath)
                 .groupby(['Subj','ImageType','Image'])
                 .agg({'Rating': 'mean', 'RT': 'mean'}).reset_index())
    all_response_data.columns = ['subject','image_type','image_name','rating','reaction_time']
    return all_response_data.groupby(['image_type','image_name'])['rating'].mean().reset_index()
        

def process_model_data(model_name, orient='wide'):
    model_data = (pd.read_csv('feature_maps/{}.csv'.format(model_name))
                  .drop(['image_type','model_layer_index'], axis = 1))
    sparsity_data = (pd.read_csv('sparsity/{}.csv'.format(model_name))
                    .drop(['image_type','model_layer_index', 'mean_activity'], axis = 1))
    model_data = pd.merge(model_data, sparsity_data, on = ['image_name','model','model_layer'])
    data_wide = pd.merge(model_data, response_data, on = 'image_name')
    model_layers = data_wide['model_layer'].unique().tolist()
    data_wide['model_layer_index'] = data_wide.apply(lambda x: model_layers.index(x['model_layer']) + 1, axis = 1)
    id_columns = ['image_name','image_type','model', 'model_layer', 'model_layer_index', 'rating']
    data_wide = data_wide[id_columns + [col for col in data_wide.columns.to_list() if col not in id_columns]]
    data_long = pd.melt(data_wide, id_vars=id_columns, 
                var_name = 'metric', value_name='value')
    
    if orient == 'wide':
        return(data_wide)
    if orient == 'long':
        return(data_long)
    
def process_corr_data(data_wide, orient='long'):
    model_layers = data_wide['model_layer'].unique().tolist()
    id_columns = ['model','image_type','model_layer', 'model_layer_index']
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
    
    def pearson_r2_score(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]**2

    scoring_metrics = {'ev_score': ev_score, 'pearson_r2': pearson_r2_score}

    model_csvs = glob('feature_maps/*.csv')
    target_models = [csv.split('/')[1].split('.')[0] for csv in model_csvs]
    #target_models = ['alexnet','vgg11','vgg13','vgg16','vgg19','resnet18','resnet34']

    output_file = 'stepwise_regressions.csv'
    if os.path.exists(output_file):
        reg_results = pd.read_csv(output_file)

    if not os.path.exists(output_file):
        results_dflist = []
        iterator = tqdm(target_models)
        for target_model in iterator:
            iterator.set_description(target_model) 
            data_wide = process_model_data(target_model)
            model_layers = data_wide['model_layer'].unique()
            data_long = process_model_data(target_model, orient='long')
                
            #for metric in tqdm(data_long['metric'].unique(), leave=False):
            for metric in tqdm(['mean_activity','sparseness'], leave=False):
                temp_path = 'incoming/combo/{}'.format(target_model)
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
                            regression = RidgeCV(alphas=[1.0], store_cv_values=True, scoring='r2').fit(X,y)
                            y_pred = regression.cv_values_.squeeze()
                            for scoring_metric in scoring_metrics:
                                score = scoring_metrics[scoring_metric](y, y_pred)
                                results_dictlist.append({'model': target_model, 'image_type': image_type, 'metric': metric,
                                                         'score_type': scoring_metric, 'score': score,
                                                         'model_layer': model_layer,
                                                         'model_depth': len(model_layers),
                                                         'model_layer_index': model_layer_index + 1,
                                                         'model_layer_depth': (model_layer_index + 1) / len(model_layers)})
                
                    incoming_results = pd.DataFrame(results_dictlist)
                    incoming_results.to_csv(temp_file, index = None)
                    results_dflist.append(incoming_results)

        results = pd.concat(results_dflist)
        results.to_csv(output_file, index = None)
