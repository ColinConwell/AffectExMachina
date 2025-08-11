from warnings import filterwarnings
filterwarnings("ignore")

from feature_analysis3 import *
from bootstrapping import *

from torchvision.datasets.utils import download_url
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

seer_model_weights_root = 'https://dl.fbaipublicfiles.com/vissl/model_zoo/'

seer_model_weights = {
    'RegNet-32Gf-SEER': 'seer_regnet32d/seer_regnet32gf_model_iteration244000.torch',
    'RegNet-32Gf-SEER-INFT': 'seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-64Gf-SEER': 'seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch',
    'RegNet-64Gf-SEER-INFT': 'seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-128Gf-SEER': ('swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/' + 
                          'model_final_checkpoint_phase0.torch'),
    'RegNet-128Gf-SEER-INFT': 'seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-256Gf-SEER': ('swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k' +
                          '_apex_syncBN64_warmup8k/' + 'model_final_checkpoint_phase0.torch'),
    'RegNet-256Gf-SEER-INFT': 'seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch'
}

seer_model_weights = {key: seer_model_weights_root + value for (key, value) in seer_model_weights.items()}

seer_weight_paths = {key: os.path.join('custom/vissl_weights', key + '.torch')
                                       for (key, value) in seer_model_weights.items()}

seer_config_names = {
    'RegNet-32Gf-SEER': 'regnet32Gf',
    'RegNet-32Gf-SEER-INFT': 'regnet32Gf',
    'RegNet-64Gf-SEER': 'regnet64Gf',
    'RegNet-64Gf-SEER-INFT': 'regnet64Gf',
    'RegNet-128Gf-SEER': 'regnet128Gf',
    'RegNet-128Gf-SEER-INFT': 'regnet128Gf',
    'RegNet-256Gf-SEER': 'regnet256Gf_1',
    'RegNet-256Gf-SEER-INFT': 'regnet256Gf_1',
}



from torchvision.datasets.utils import download_url

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Feature Extraction for Aesthetics Dataset')
    parser.add_argument('--model_name', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--imageset', type=str, required=False, default='oasis',
                        help='imageset to use for extracting features')
    parser.add_argument('--output_type', type=str, required=False, default='csv',
                        help='format of output_file (csv or parquet)')
    parser.add_argument('--cuda_device', required=False, default='7',
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_name = args.model_name
    imageset = args.imageset
    output_type = args.output_type
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    train_type = 'seer'
    model_string = '_'.join([model_name, train_type])
    model_option = {'model_name': model_name,
                    'train_type': train_type}
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now processing {} with {} on {}...'.format(model_name, device_name, imageset))

    analyses = ['bootstrapping']
    output_files = prepare_output_files('incoming', output_type, analyses, imageset, model_string)
    
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        
        if not os.path.exists('custom/vissl_weights/{}.torch'.format(model_name)):
            download_url(seer_model_weights[model_name], 'custom/vissl_weights', model_name + '.torch')

        model_config = seer_config_names[model_name]

        cfg = ['config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear', 
               '+config/benchmark/linear_image_classification/imagenet1k/models={}'.format(model_config), 
               'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=custom/vissl_weights/{}.torch'.format(model_name)]

        cfg = compose_hydra_configuration(cfg)
        _, cfg = convert_to_attrdict(cfg)
                              
        model = build_model(cfg.MODEL, cfg.OPTIMIZER)
        model = model.eval()
        
        weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

        init_model_from_consolidated_weights(config=cfg, model=model, 
                                             state_dict=weights, 
                                             skip_layers = [],
                                             state_dict_key_name="classy_state_dict")
                              
        image_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        image_data = load_image_data(imageset)

        stimulus_loader = get_stimulus_loader(image_data.image_path, image_transforms)
        
        response_data = load_response_data(imageset, 'response')
        response_data = copy(image_data).merge(response_data, on = 'image_name')
        
        if 'metrics' in output_files and not os.path.exists(output_files['metrics']):
            
            stimulus_features = get_all_feature_maps(model, inputs = stimulus_loader)
            metric_data = get_feature_metrics(model_option, stimulus_features)
            metric_data.to_csv(output_files['metrics'], index = None)
        
        if 'reg_redux' in output_files and not os.path.exists(output_files['reg_redux']):
            
            stimulus_features = get_all_feature_maps(model, inputs = stimulus_loader)
            stimulus_features = get_feature_map_srps(stimulus_features, delete_originals = True)
            
            reg_results = get_regression_results(model_option, stimulus_features, response_data, alpha_values = [1000])
            reg_results.to_csv(output_files['reg_redux'], index = None)
            
        if 'bootstrapping' in output_files and not os.path.exists(output_files['bootstrapping']):
            
            bootstrap_data = pd.read_csv('response/{}_bootstraps.csv'.format(imageset))
        
            target_layers = pd.read_csv('superlative_layers.csv').set_index('model_string').to_dict(orient='index')
            target_layer = target_layers[model_string]['model_layer']
            
            stimulus_features = get_all_feature_maps(model, inputs = stimulus_loader, layers_to_retain = [target_layer])
            stimulus_features = get_feature_map_srps(stimulus_features)

            bootstrap_results = get_bootstrap_regression_results(model_option, stimulus_features, response_data,
                                                                 bootstrap_data, alpha_values = [1000])

            if output_type == 'csv':
                bootstrap_results.to_csv(output_files['bootstrapping'], index = None)
            if output_type == 'parquet':
                bootstrap_results.to_parquet(output_files['bootstrapping'], index = None)

