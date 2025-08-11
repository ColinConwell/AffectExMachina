from dataset import *
import numpy as np
import argparse
from tqdm.auto import tqdm as tqdm

def get_bootstrap_sample(response_data, image_data, measurement):
    image_type_reference = response_data[['image_name','image_type']].drop_duplicates()
    subject_data = (response_data[['subject',measurement,'image_name']]
                .pivot(index = ['subject'], columns = 'image_name', values = measurement))
    randlist = pd.DataFrame(index=np.random.choice(subject_data.index.unique(), size=subject_data.shape[0]))
    bootstrap_sample = subject_data.merge(randlist, left_index=True, right_index=True, how='right')
    bootstrap_response_data = (bootstrap_sample.mean(axis = 0).reset_index()
                               .rename(columns = {0: 'rating', 'index': 'image_name'}))
    bootstrap_response_data['measurement'] = measurement
    bootstrap_response_data = bootstrap_response_data.merge(image_type_reference, on = 'image_name')
    
    return image_data.merge(bootstrap_response_data, on = 'image_name')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get bootstrap samples of subject responses')
    parser.add_argument('--n_bootstraps', type=int, required=False, default=10000,
                        help='imageset to use for extracting features')
    parser.add_argument('--save_wide', type=bool, required=False, default=True,
                        help='imageset to use for extracting features')
    parser.add_argument('--imageset', type=str, required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', type=str, required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--cuda_device', required=False, default='7',
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    n_bootstraps = args.n_bootstraps
    save_wide = args.save_wide
    imageset = args.imageset
    output_type = args.output_type
    cuda_device = args.cuda_device
    
    output_file = 'response/{}_bootstraps2.{}'.format(imageset, output_type)
    print('Now saving {} bootstrapped samples from {} to {}'.format(n_bootstraps, imageset, output_file))
    
    response_data = load_response_data(imageset, average = False)
    image_data = load_image_data(imageset)
    
    bootstrap_list = []
    measurements = [col for col in response_data.columns if col in ['arousal', 'valence', 'beauty']]
    
    for measurement in measurements:
        for i in tqdm(range(n_bootstraps)):
            bootstrap_sample = get_bootstrap_sample(response_data, image_data, measurement)
            bootstrap_sample['bootstrap_id'] = i+1
            bootstrap_list.append(bootstrap_sample)
                
    bootstraps = pd.concat(bootstrap_list)
    
    if save_wide:
        bootstraps = (bootstraps[['image_name','measurement','rating','bootstrap_id']]
                      .pivot(index = ['image_name','measurement'], 
                             columns = 'bootstrap_id', values = 'rating').reset_index())
        
    bootstraps = np.round(bootstraps, 5)
        
    if output_type == 'csv':
        bootstraps.to_csv(output_file, index = None)
    if output_type == 'parquet':
        bootstraps.to_parquet(output_file, index = None)
        
    
    
    
    
        
            
            
            