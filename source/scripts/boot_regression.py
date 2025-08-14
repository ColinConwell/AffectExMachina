from .feature_analysis import *
from ..model_opts.mapping_methods import *

from pathlib import Path

_THIS_DIRPATH = Path(__file__).parent
_LAYER_LOOKUP = _THIS_DIRPATH / 'results' / 'superlative_layers.csv'

def get_bootstrap_regression_results(model_option, stimulus_features, response_data, bootstrap_data,
                                     alpha_values = np.logspace(-1,5,25).tolist()):
    
    image_reference = response_data[['image_name','image_type']].drop_duplicates()
    
    scoring_metrics = ['explained_variance', 'pearson_r']
    
    model_name = model_option['model_name']
    train_type = model_option['train_type']
        
    scoresheets = []
    for model_layer_index, model_layer in enumerate(tqdm(stimulus_features, desc = 'Regression (Layer)')):
        target_features = stimulus_features[model_layer]
        if isinstance(stimulus_features[model_layer], torch.Tensor):
            target_features = target_features.numpy()

        measurements = [column for column in response_data.columns if column in ['arousal','beauty','valence']]
        for measurement in measurements:
            bootstrap_data_sub = bootstrap_data[bootstrap_data['measurement'] == measurement].reset_index(drop=True)
            for image_type in image_reference['image_type'].unique().tolist() + ['Combo']:
                if image_type != 'Combo':
                    image_indices = image_reference[image_reference['image_type'] == image_type].index.to_numpy()
                if image_type == 'Combo':
                    response_data_sub = response_data
                    image_indices = image_reference.index.to_numpy()

                y = bootstrap_data_sub.iloc[image_indices,2:].to_numpy()
                X = scale(target_features[image_indices,:])
                regression = RidgeCV(alphas=alpha_values, store_cv_values=True,
                                     scoring='explained_variance').fit(X,y)
                
                n_bootstraps = bootstrap_data_sub.shape[1] - 2
                for alpha_value in alpha_values:
                    y_pred = regression.cv_values_[:, :, alpha_values.index(alpha_value)]

                    for score_type in scoring_metrics:
                        ridge_gcv_score = score_func(y, y_pred, score_type)
                        
                        bootstrap_ids = list(range(n_bootstraps))
                        scoresheet = pd.DataFrame({'model': model_name, 'train_type': train_type, 
                                                   'model_layer_index': model_layer_index+1,
                                                   'model_layer': model_layer,
                                                   'measurement': measurement,
                                                   'image_type': image_type,
                                                   'score_type': score_type,
                                                   'bootstrap_ids': bootstrap_ids,
                                                   'score': ridge_gcv_score, 
                                                   'alpha': alpha_value})
                        
                        scoresheets.append(scoresheet)
                        
    return pd.concat(scoresheets) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Bootstrapping for Feature Regressions')
    parser.add_argument('--model_string', type=str, required=True, 
                        help='name of deep net model to load')
    parser.add_argument('--imageset', required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', required=False, default='parquet',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--output_dir', required=False, default='incoming', 
                        help='destination for output files')
    parser.add_argument('--cuda_device', required=False, default='6',
                        help='target cuda device for gpu compute')
    parser.add_argument('--layer_lookup_path', type=str, 
                        required=False, default=_LAYER_LOOKUP,
                        help='path to layer lookup table')
    
    args = parser.parse_args()
    model_string = args.model_string
    imageset = args.imageset
    output_type = args.output_type
    output_dir = args.output_dir
    layer_lookup_path = args.layer_lookup_path
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now processing {} with {} on {}...'.format(model_string, device_name, imageset))
    
    model_option = get_model_options()[model_string]
    analyses = ['bootstrapping']

    output_files = prepare_output_files(output_dir, output_type, analyses, imageset, model_string) 
    
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        
        image_data = load_image_data(imageset)
        image_transforms = get_recommended_transforms(model_string)

        stimulus_loader = get_stimulus_loader(image_data.image_path, image_transforms)
        
        response_data = load_response_data(imageset, 'response')
        response_data = copy(image_data).merge(response_data, on = 'image_name')
        
        bootstrap_data = pd.read_csv('response/{}_bootstraps.csv'.format(imageset))
        
        target_layers = pd.read_csv(layer_lookup_path).set_index('model_string').to_dict(orient='index')
        target_layer = target_layers[model_string]['model_layer']
        
        feature_maps = get_all_feature_maps(model_string, stimulus_loader, layers_to_retain = [target_layer])
        stimulus_features = get_feature_map_srps(feature_maps)
        
        bootstrap_results = get_bootstrap_regression_results(model_option, stimulus_features, response_data,
                                                             bootstrap_data, alpha_values = [1000])
        
        if output_type == 'csv':
            bootstrap_results.to_csv(output_files['bootstrapping'], index = None)
        if output_type == 'parquet':
            bootstrap_results.to_parquet(output_files['bootstrapping'], index = None)
            
        
        
