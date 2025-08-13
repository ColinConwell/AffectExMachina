from .feature_analysis import *

def cross_ridge_regression(X, y, train_indices, test_indices, train_y='beauty', test_y='beauty', alpha_values=[1000]):

    X_train = X[train_indices]
    y_train = y[train_indices][train_y]

    X_test = X[test_indices]
    y_test = y[test_indices][test_y]

    regression = RidgeCV(alphas=alpha_values, store_cv_values=True,
                         scoring='explained_variance').fit(X_train,y_train)

    y_train_pred = regression.cv_values_[:, alpha_values.index(regression.alpha_)]
    
    iid_score = pearson_r_score(y_train, y_train_pred)
    ood_score = pearson_r_score(regression.predict(X_test), y_test)

    return(regression.alpha_, iid_score, ood_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Decoding Dataset')
    parser.add_argument('--model_string', type=str, required=True,
                        help='name of deep net model to load')
    parser.add_argument('--cross_type', type = str, required=True, 
                        help='whether to cross-decode affect or image_type')
    parser.add_argument('--output_type', type=str, required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--cuda_device', required=False, default='7',
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_string = args.model_string
    cross_type = args.cross_type
    output_type = args.output_type
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now cross-decoding {} with {} on {}...'.format(cross_type, model_string, device_name))
    
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']
    
    output_dir = os.path.join('incoming', 'cross_regs', cross_type)
    file_name = '{}.{}'.format(model_string.replace('/','-'), output_type)
    output_file = os.path.join(output_dir, file_name)
    print('Saving results to {}...'.format(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_file):
        
        response_data = pd.concat([load_response_data(imageset, 'response') for imageset in ['oasis', 'vessel']])
        
        asset_dictlist = []
        for imageset in ['oasis', 'vessel']:
            root = 'images/{}/'.format(imageset)
            assets = glob(root + '*.jpg')
            for asset in assets:
                imgstr = asset.split('/')[-1]
                row = {'image_name': imgstr,
                       'image_path': asset,
                       'dataset': imageset}
                asset_dictlist.append(row)
        image_data = (pd.DataFrame(asset_dictlist)
                      .sort_values(by='image_name', ignore_index=True))
        
        response_data = copy(image_data).merge(response_data, on = 'image_name')
        
        image_transforms = get_recommended_transforms(model_string)
        stimulus_loader = DataLoader(dataset=StimulusSet(image_data.image_path, image_transforms), batch_size=64)
        
        target_layers = pd.read_csv('superlative_layers.csv').set_index('model_string').to_dict(orient='index')
        target_layer = target_layers[model_string]['model_layer']
        
        feature_maps = get_all_feature_maps(model_string, stimulus_loader, layers_to_retain = [target_layer])
        stimulus_features = get_feature_map_srps(feature_maps)
        
        target_features = scale(stimulus_features[target_layer])
        
        if cross_type == 'image_type':
            data_opts = response_data['image_type'].unique().tolist() + ['oasis_combo','vessel_combo']

            cross_decode_dictlist = []
            train_data_iterator = tqdm(data_opts, leave = False)
            for train_data in train_data_iterator:
                train_data_iterator.set_description(train_data)
                test_data_iterator = tqdm(data_opts, leave = False)
                for test_data in test_data_iterator:
                    test_data_iterator.set_description(test_data)
                    if 'combo' in train_data:
                        train_indices = (response_data['dataset'] == train_data.split('_')[0])
                    if 'combo' in test_data:
                        test_indices = (response_data['dataset'] == test_data.split('_')[0])
                    if 'combo' not in train_data:
                        train_indices = (response_data['image_type'] == train_data)
                    if 'combo' not in test_data:
                        test_indices = (response_data['image_type'] == test_data)

                    alpha, base_score, cross_score = cross_ridge_regression(target_features, response_data, 
                                                                            train_indices, test_indices)

                    cross_decode_dictlist.append({'model': model_name, 'train_type': train_type, 
                                                  'model_layer': target_layer, 'alpha': alpha,
                                                  'train_data': train_data, 
                                                  'test_data': test_data, 
                                                  'base_score': base_score, 
                                                  'cross_score': cross_score})

            cross_decoding = pd.DataFrame(cross_decode_dictlist)
                
        if cross_type == 'affect': 
            data_opts = [opt for opt in response_data['image_type'].unique().tolist() + ['oasis_combo'] 
                         if opt in ['Scene','Animal','Object','Person','oasis_combo']]


            cross_decode_dictlist = []
            train_data_iterator = tqdm(data_opts, leave = False)
            for train_data in train_data_iterator:
                train_data_iterator.set_description(train_data)
                test_data_iterator = tqdm(data_opts, leave = False)
                for test_data in test_data_iterator:
                    test_data_iterator.set_description(test_data)
                    if 'combo' in train_data:
                        train_indices = (response_data['dataset'] == train_data.split('_')[0])
                    if 'combo' in test_data:
                        test_indices = (response_data['dataset'] == test_data.split('_')[0])
                    if 'combo' not in train_data:
                        train_indices = (response_data['image_type'] == train_data)
                    if 'combo' not in test_data:
                        test_indices = (response_data['image_type'] == test_data)

                    #print(train_indices.shape, test_indices.shape)

                    for train_affect in ['beauty','arousal', 'valence']:
                        for test_affect in ['beauty','arousal', 'valence']:

                            alpha, base_score, cross_score = cross_ridge_regression(target_features, response_data,
                                                                                    train_indices, test_indices,
                                                                                    train_y = train_affect, 
                                                                                    test_y = test_affect)

                            cross_decode_dictlist.append({'model': model_name, 'train_type': train_type, 
                                                          'model_layer': target_layer, 'alpha': alpha,
                                                          'train_data': train_data, 'test_data': test_data,
                                                          'train_y': train_affect, 'test_y': test_affect,
                                                          'train': train_data + '_' + train_affect,
                                                          'test': test_data + '_' + test_affect,
                                                          'base_score': base_score, 'cross_score': cross_score})

            cross_decoding = pd.DataFrame(cross_decode_dictlist)
        
        if output_type == 'csv':
            cross_decoding.to_csv(output_file, index = None)
        if output_type == 'parquet':
            cross_decoding.to_parquet(output_file, index = None)
        
        
        
        