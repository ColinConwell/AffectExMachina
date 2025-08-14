import os, pandas as pd
from glob import glob

def load_response_data(imageset, response_dir = 'response', average = True):
    if average == True:
        response_data = pd.read_csv(os.path.join(response_dir, '{}_means_per_image.csv').format(imageset))
        if imageset == 'oasis':
            response_data = response_data.rename(columns={'category': 'image_type'})
            response_data['image_name'] = response_data['theme'] + '.jpg'
            response_data = response_data.drop(['item','theme'], axis = 1)
        if imageset == 'vessel':
            response_data = response_data.rename(columns={'rating': 'beauty'})
    if average == False:
        if imageset == 'oasis':
            aenne_data_path = '{}/aenne_subject_data.csv'.format(response_dir)
            oasis_data_path = '{}/oasis_subject_data.csv'.format(response_dir)
            aenne_subject_data = pd.read_csv(aenne_data_path).drop(['arousal','valence'], axis = 1)
            oasis_subject_data = pd.read_csv(oasis_data_path)
            subject_data_combo = pd.concat([oasis_subject_data, aenne_subject_data], axis = 0)
            response_data = subject_data_combo.rename(columns={'category': 'image_type'})
            response_data['image_name'] = response_data['theme'] + '.jpg'
            response_data.image_name = response_data.image_name.str.replace(' ', '')
            response_data = response_data.drop(['item','theme'], axis = 1)
        if imageset == 'vessel':
            response_data = (pd.read_csv('response/vessel_subject_data.csv')
                 .groupby(['Subj','ImageType','Image'])
                .agg({'Rating': 'mean'}).reset_index())
            response_data.columns = ['subject','image_type','image_name','beauty']
            
    return response_data

def load_image_data(imageset, image_dir = 'images'):
    root = '{}/{}/'.format(image_dir, imageset)
    assets = glob(root + '*.jpg')
    asset_dictlist = []
    for asset in assets:
        imgstr = asset.split('/')[-1]
        row = {'image_name': imgstr,
               'image_path': asset,
               'imageset': imageset}
        asset_dictlist.append(row)
    image_data = (pd.DataFrame(asset_dictlist)
                  .sort_values(by='image_name', ignore_index=True))
    
    return image_data
    
def load_combined_response_data(response_dir = 'response', average = True):
        return pd.concat([load_response_data('vessel', response_dir, average).assign(imageset = 'vessel'),
                          load_response_data('oasis', response_dir, average).assign(imageset = 'oasis')], axis = 0)
    
def load_combined_image_data(image_dir = 'images'):
    return pd.concat([load_image_data('oasis', image_dir), load_image_data('vessel', image_dir)], axis = 0)
    
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