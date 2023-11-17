BEST_CONFIGS = {
    "node_classification": {
        'RGCN': {
            'general': {
                'seed': 0, 'patience': 40, 'validation': True, 'evaluation_metric': 'acc'
            },
            'aifb': {
                'lr': 0.005, 'weight_decay': 0, 'max_epoch': 100,
                'hidden_dim': 32, 'n_bases': -1, 'num_layers': 2, 'batch_size': 126, 'dropout': 0,
                'mini_batch_flag': False, 'validation': True
            },
            'mutag': {
                'lr': 0.005, 'weight_decay': 0.0005, 'max_epoch': 50,
                'hidden_dim': 32, 'n_bases': 30, 'num_layers': 2, 'batch_size': 50, 'fanout': 4, 'dropout': 0.5,
                'mini_batch_flag': False, 'validation': True
            },
            'bgs': {
                'lr': 0.005, 'weight_decay': 0.0005, 'max_epoch': 50,
                'hidden_dim': 16, 'n_bases': 40, 'num_layers': 3, 'batch_size': 126, 'fanout': 4, 'dropout': 0.1,
                'mini_batch_flag': True, 'validation': True
            }
        }
    }
    
}

def get_configs(dataset,task = "node_classification", model = "RGCN"):
    general = BEST_CONFIGS[task][model]['general']
    data_config = BEST_CONFIGS[task][model][dataset]
    return data_config