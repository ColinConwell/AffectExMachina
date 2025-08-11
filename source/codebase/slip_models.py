from warnings import filterwarnings
filterwarnings("ignore")

from feature_analysis3 import *
from bootstrapping import *

sys.path.append('custom/slip_codebase')
import models
import utils
from tokenizer import SimpleTokenizer

slip_model_weights = {
    'ViT-S-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_small_25ep.pt',
    'ViT-S-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_small_25ep.pt',
    'ViT-S-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_small_25ep.pt',
    'ViT-S-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_small_100ep.pt',
    'ViT-B-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_base_25ep.pt',
    'ViT-B-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_base_25ep.pt',
    'ViT-B-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_base_25ep.pt',
    'ViT-B-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt',
    'ViT-L-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_large_25ep.pt',
    'ViT-L-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_large_25ep.pt',
    'ViT-L-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_large_25ep.pt',
    'ViT-L-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_large_100ep.pt',
    'ViT-L-CLIP-CC12M': 'https://dl.fbaipublicfiles.com/slip/clip_base_cc12m_35ep.pt',
    'ViT-L-SLIP-CC12M': 'https://dl.fbaipublicfiles.com/slip/slip_base_cc12m_35ep.pt',
}

slip_weight_paths = {key: os.path.join('custom/slip_weights', value.split('/')[-1])
                                       for (key, value) in slip_model_weights.items()}

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
    
    train_type = 'slip'
    model_string = '_'.join([model_name, train_type])
    model_option = {'model_name': model_name,
                    'train_type': train_type}
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Now processing {} with {} on {}...'.format(model_name, device_name, imageset))

    analyses = ['bootstrapping']
    output_files = prepare_output_files('incoming', output_type, analyses, imageset, model_string)
    
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        
        if not os.path.exists(slip_weight_paths[model_name]):
            download_url(slip_model_weights[model_name], 'custom/slip_weights')
        
        ckpt_path = slip_weight_paths[model_name]
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        # create model
        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))
        model = getattr(models, old_args.model)(rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
        
        image_transforms = transforms.Compose([
          transforms.Resize((224,224)),
          lambda x: x.convert('RGB'),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_data = load_image_data(imageset)

        stimulus_loader = get_stimulus_loader(image_data.image_path, image_transforms)
        
        response_data = load_response_data(imageset, 'response')
        response_data = copy(image_data).merge(response_data, on = 'image_name')
        
        if 'metrics' in output_files and not os.path.exists(output_files['metrics']):
            
            metric_data = get_feature_metrics(model_option, stimulus_features)
            metric_data.to_csv(output_files['metrics'], index = None)
        
        if 'reg_redux' in output_files and not os.path.exists(output_files['reg_redux']):
            
            stimulus_features = get_all_feature_maps(model.visual, inputs = stimulus_loader)
            stimulus_features = get_feature_map_srps(stimulus_features, delete_originals = True)
            
            reg_results = get_regression_results(model_option, stimulus_features, response_data, alpha_values = [1000])
            reg_results.to_csv(output_files['reg_redux'], index = None)
            
        if 'bootstrapping' in output_files and not os.path.exists(output_files['bootstrapping']):
            
            bootstrap_data = pd.read_csv('response/{}_bootstraps.csv'.format(imageset))
        
            target_layers = pd.read_csv('superlative_layers.csv').set_index('model_string').to_dict(orient='index')
            target_layer = target_layers[model_string]['model_layer']
            
            stimulus_features = get_all_feature_maps(model.visual, inputs = stimulus_loader, layers_to_retain = [target_layer])
            stimulus_features = get_feature_map_srps(stimulus_features)

            bootstrap_results = get_bootstrap_regression_results(model_option, stimulus_features, response_data,
                                                                 bootstrap_data, alpha_values = [1000])

            if output_type == 'csv':
                bootstrap_results.to_csv(output_files['bootstrapping'], index = None)
            if output_type == 'parquet':
                bootstrap_results.to_parquet(output_files['bootstrapping'], index = None)

