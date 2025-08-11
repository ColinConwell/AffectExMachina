from feature_analysis3 import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Decoding Dataset')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--imageset', type=str, required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', type=str, required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--cuda_device', required=False, default='7',
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
    
    output_dir = os.path.join('incoming', 'subject_regs', imageset)
    file_name = '{}.{}'.format(model_string.replace('/','-'), output_type)
    output_file = os.path.join(output_dir, file_name)
    print('Saving results to {}...'.format(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(output_file):
        
        response_data = load_response_data(imageset, average = False)
        image_data = load_image_data(imageset)
        
        image_transforms = get_recommended_transforms(model_string)
        stimulus_loader = DataLoader(dataset=StimulusSet(image_data.image_path, image_transforms), batch_size=64)
        
        target_layers = pd.read_csv('superlative_layers.csv').set_index('model_string').to_dict(orient='index')
        target_layer = target_layers[model_string]['model_layer']
        
        feature_maps = get_all_feature_maps(model_string, stimulus_loader, layers_to_retain = [target_layer])
        stimulus_features = get_feature_map_srps(feature_maps)
        
        target_features = stimulus_features[target_layer]
        if isinstance(target_features, torch.Tensor):
            target_features = target_features.numpy()

        score_dictlist = []
        data_i = response_data.copy().merge(image_data, on = ['image_name'])
        for measurement in tqdm([col for col in response_data.columns if col in ['arousal','beauty','valence']]):
            data_i_sub1 = data_i[['subject', 'image_name', 'image_type', measurement]]
            for image_type in tqdm(data_i['image_type'].unique().tolist() + ['Combo'], leave = False):
                if image_type != 'Combo':
                    data_i_subset = data_i_sub1[data_i_sub1['image_type'] == image_type]
                if image_type == 'Combo':
                    data_i_subset = data_i_sub1
                for subject in tqdm(data_i_subset['subject'].unique(), leave = False):
                    group_data_i = (data_i_subset[data_i_subset['subject'] != subject].groupby('image_name')[measurement]
                                    .mean().reset_index()[measurement]).to_numpy()
                    response_data_i = data_i_subset[data_i_subset['subject'] == subject][measurement].to_numpy()
                    item_indices = np.argwhere(~np.isnan(response_data_i)).flatten()
                    if len(item_indices) > 10:
                        y, y_group = response_data_i[item_indices], group_data_i[item_indices]

                        X = scale(target_features[item_indices,:])

                        alpha_values = [1000]
                        regression = RidgeCV(alphas=alpha_values, store_cv_values=True,
                                             scoring='explained_variance').fit(X,y)

                        ridge_gcv_score, ridge_gcv_alpha = regression.best_score_, regression.alpha_
                        y_pred = regression.cv_values_[:, alpha_values.index(ridge_gcv_alpha)]

                        for alpha_value in alpha_values:
                            y_pred = regression.cv_values_[:, alpha_values.index(alpha_value)]

                            for score_type in scoring_metrics:
                                ridge_gcv_score = scoring_metrics[score_type](y, y_pred)

                                score_dictlist.append({'model': model_name, 'train_type': train_type, 
                                                       'model_layer': target_layer,
                                                       'subject': subject, 
                                                       'measurement': measurement,
                                                       'image_type': image_type,
                                                       'image_count': len(item_indices),
                                                       'score_type': score_type,
                                                       'score': ridge_gcv_score, 
                                                       'alpha': regression.alpha_})
                            
        subject_reg_data = pd.DataFrame(score_dictlist)
        
        if output_type == 'csv':
            subject_reg_data.to_csv(output_file, index = None)
        if output_type == 'parquet':
            subject_reg_data.to_parquet(output_file, index = None)
        