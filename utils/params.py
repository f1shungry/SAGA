import argparse
import torch
import sys


def get_params():
    """
    Unified parameter function that returns default parameters based on dataset name
    Users can override any parameter via command line
    """
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name, e.g.: ACM1-ACM2, ACM2-ACM1, CN-US, US-CN, JP-CN, CN-JP, DE-FR, FR-DE, RU-US, US-RU, CN-DE, DE-CN')
    parser.add_argument('--load_parameters', default=False, action='store_true',
                        help='Whether to load saved parameters')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--batchsize', type=int, default=None,
                        help='Batch size (if not specified, automatically selected based on dataset)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--nb_epochs', type=int, default=None,
                        help='Number of epochs (if not specified, automatically selected based on dataset)')

    # Optimizer parameters
    parser.add_argument('--l2_coef', type=float, default=1e-4,
                        help='L2 regularization coefficient')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter')

    # Model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.0,
                        help='Filter alpha parameter')

    # Parse known args first
    args, _ = parser.parse_known_args()

    # Dataset-specific default values
    dataset_defaults = {
        'ACM1-ACM2': {
            'batchsize': 1000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'ACM2-ACM1': {
            'batchsize': 1000,
            'nb_epochs': 100,
            'dropout': 0.05,
            'seed': 0
        },
        'CN-US': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'US-CN': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'JP-CN': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'CN-JP': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'DE-FR': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'FR-DE': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'RU-US': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'US-RU': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'CN-DE': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        },
        'DE-CN': {
            'batchsize': 10000,
            'nb_epochs': 50,
            'dropout': 0.0,
            'seed': 0
        }
    }

    # Check if dataset exists
    if args.dataset not in dataset_defaults:
        available_datasets = list(dataset_defaults.keys())
        raise ValueError(f"Unknown dataset: {args.dataset}\nAvailable datasets: {available_datasets}")

    # Get defaults for current dataset
    defaults = dataset_defaults[args.dataset]

    # Use default values if user didn't specify
    if args.batchsize is None:
        args.batchsize = defaults['batchsize']
    if args.nb_epochs is None:
        args.nb_epochs = defaults['nb_epochs']
    if args.dropout == 0.0 and 'dropout' in defaults:
        args.dropout = defaults['dropout']
    if args.seed == 0 and 'seed' in defaults:
        args.seed = defaults['seed']

    return args