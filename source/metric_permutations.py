from warnings import filterwarnings
filterwarnings("ignore")

import os, sys, json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tqdm

from scipy.stats import pearsonr
from glob import glob

sys.path.append('../mouseland/model_opts')
from model_options import *
model_options = get_model_options()

import numba

NAN = float("nan")

@numba.njit(nogil=True)
def _any_nans(a):
    for x in a:
        if np.isnan(x): return True
    return False

@numba.jit
def any_nans(a):
    if not a.dtype.kind=='f': return False
    return _any_nans(a.flat)

from processing import *

response_data = {'vessel': load_response_data('vessel'), 'oasis': load_response_data('oasis')}

def process_metric_data(model_string, dataset, orient='wide'):
    model_data = (pd.read_csv('metrics/{}/{}.csv'.format(dataset,model_string)))
    model_data['dataset'] = dataset
    if 'image' in model_data.columns:
        model_data = model_data.rename(columns={'image': 'image_name'})
    
    data_wide = pd.merge(model_data, response_data[dataset], on = 'image_name')
    data_wide['model_layer_depth'] = (data_wide['model_layer_index'] / 
                                      data_wide['model_layer'].nunique())
    
    id_columns = ['dataset','image_name','image_type','model','train_type',
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

if __name__ == "__main__":
    
    model_csvs = glob('metrics/oasis/*.csv')
    target_models = [csv.split('/')[-1].split('.')[0] for csv in model_csvs]
    target_models = [model for model in target_models if 'googlenet' not in model]

    #target_models = ['alexnet_imagenet', 'alexnet_random']

    output_file = 'results/metric_permuting.csv'
    if os.path.exists(output_file):
        perm_results = pd.read_csv(output_file)

    if not os.path.exists(output_file):
        results_dflist = []
        iterator = tqdm(sorted(target_models), leave = False)
        for target_model in iterator:
            iterator.set_description(target_model)
            results_dictlist = []
            temp_path = 'incoming/permutations'
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            temp_file = '{}/{}.csv'.format(temp_path, target_model)
            if os.path.exists(temp_file):
                results_dflist.append(pd.read_csv(temp_file))
            
            if not os.path.exists(temp_file):
                for dataset in ['vessel','oasis']:
                    data_wide = process_metric_data(target_model, dataset)
                    model_name = model_options[target_model]['model_name']
                    train_type = model_options[target_model]['train_type']
                    model_layers = data_wide['model_layer'].unique()
                    for measurement in data_wide['measurement'].unique():
                            for image_type in data_wide['image_type'].unique().tolist() + ['Combo']:
                                for metric in ['mean_activity','sparseness']:
                                    data_i = data_wide[(data_wide['image_type'] == image_type) & 
                                                       (data_wide['measurement'] == measurement)]
                                    y = data_i[(data_i['model_layer']==model_layers[0])]['rating'].to_numpy()
                                    X = np.stack([data_i[(data_i['model_layer']==model_layer)][metric].to_numpy() 
                                                  for model_layer in model_layers], axis = 1)

                                    actual_max = max([abs(pearsonr(x, y)[0]) for x in X.transpose()
                                                      if not any_nans(x)])

                                    permuted_max_corrs = []
                                    for i in range(1000):
                                        permuted_corrs = [abs(pearsonr(np.random.permutation(x), y)[0]) 
                                                          for x in X.transpose() if not any_nans(x)]
                                        permuted_max_corrs.append(max(permuted_corrs))

                                    permuted_lqt = np.quantile(permuted_max_corrs, 0.025)
                                    permuted_uqt = np.quantile(permuted_max_corrs, 0.975)
                                    permuted_pvalue = (len([corr for corr in permuted_max_corrs if corr >= actual_max]) + 1) / 1001

                                    results_dictlist.append({'model': model_name, 'train_type': train_type, 
                                                             'dataset': dataset, 'image_type': image_type, 
                                                             'metric': metric, 'measurement': measurement,
                                                             'model_depth': len(model_layers),
                                                             'corr_max_score': actual_max,
                                                             'corr_lower_ci': permuted_lqt,
                                                             'corr_upper_ci': permuted_uqt,
                                                             'corr_p_value': permuted_pvalue})


                incoming_results = pd.DataFrame(results_dictlist)
                incoming_results.to_csv(temp_file, index = None)
        
        perm_results = pd.concat(results_dflist)
        perm_reseults['corr_p_adj'] = pg.multicomp(perm_results['corr_p_value'].to_numpy(), 
                                                   alpha = 0.05, method = 'fdr')
        
        perm_results.to_csv(output_file, index = None)