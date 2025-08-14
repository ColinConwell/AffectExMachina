import pandas as pd
from ..dataset import *

def max_transform(df, group_vars, measure_var = 'score', transform = max, deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = [group_vars]
    
    max_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(max) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        max_df = max_df[~max_df.duplicated(group_vars + [measure_var])]
        
    return max_df

def min_transform(df, group_vars, measure_var = 'score', transform = max, deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = [group_vars]
    
    min_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(min) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        min_df = min_df[~min_df.duplicated(group_vars + [measure_var])]
        
    return min_df

def process_metric_data(model_string, dataset, orient='wide'):
    model_data = (pd.read_csv('metrics/{}/{}.csv'.format(dataset,model_string)))
    model_data['dataset'] = dataset
    if 'image' in model_data.columns:
        model_data = model_data.rename(columns={'image': 'image_name'})
    
    data_wide = pd.merge(model_data, load_response_data(dataset), on = 'image_name')
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

def process_corr_data(data_wide, include_combo = True, orient='long'):
    model_layers = data_wide['model_layer'].unique().tolist()
    
    id_columns = ['model','train_type','dataset','image_type','model_layer',
                  'model_layer_index','model_layer_depth', 'measurement']
    
    corr_data_wide = (data_wide.groupby(id_columns).corrwith(data_wide['rating'])
                      .reset_index().drop('rating',axis = 1))
    
    if include_combo:
        
        id_columns_ = [col for col in id_columns if col != 'image_type']
        
        corr_data_wide_ = (data_wide.groupby(id_columns_).corrwith(data_wide['rating'])
                           .reset_index().drop('rating',axis = 1))
        corr_data_wide_['image_type'] = 'Combo'
        
        corr_data_wide = pd.concat([corr_data_wide, corr_data_wide_])
        
    
    corr_data_long = pd.melt(corr_data_wide, id_vars = id_columns, 
                             var_name = 'metric', value_name='corr')
        
    if orient == 'wide':
        return(corr_data_wide)
    if orient == 'long':
        return(corr_data_long)
