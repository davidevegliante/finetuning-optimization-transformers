import argparse

def create_train_argparse():
    parser = argparse.ArgumentParser(description='Fine-Tuning')

    # with no default
    parser.add_argument('--model_name', type=str, required=True, help='Base Transformer model to use')
    parser.add_argument('--tuning_strategy', choices=['pooled', 'concat', 'avg'], type=str, required=True, help='Fine-Tuning strategy to apply') 

    # default
    parser.add_argument('--output_path', type=str, required=False, default='models', help='Output directory')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=False, default=4, help='Number of epochs')
    parser.add_argument('--lr', type=float, required=False, default=2e-5, help='Learning Rate')
    parser.add_argument('--fixed_seed', type=int, required=False, default=None, help='Fix the seed and set operation to be deterministic')
    parser.add_argument('--max_seq_len', type=int, required=False, default=125, help='Train example max sequence length')
    parser.add_argument('--evaluate_test_set', type=bool, required=False, default=None, help='At the end of the training phase, evaluate the model on test set')

    return parser.parse_args()


def create_benchmark_argparse():
    parser = argparse.ArgumentParser(description='')

    # with no default
    parser.add_argument('--model_name', type=str, required=True, help='Base Transformer model to use')
    parser.add_argument('--tuning_strategy', choices=['pooled', 'concat', 'avg'], type=str, required=True, help='Fine-Tuning strategy to apply') 
    parser.add_argument('--model_state_dict', type=str, required=True, help='State dict path')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')

    return parser.parse_args()
