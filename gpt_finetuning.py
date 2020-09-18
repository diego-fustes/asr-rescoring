"""Fine-tuning GPT model with a domain dataset"""
import argparse
import logging
from distutils.util import strtobool

from simpletransformers.language_modeling import LanguageModelingModel

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("train_path", type=str, default="data/train.tokens",
                    help="Path to training dataset")
parser.add_argument("validation_path", type=str, default="data/val.tokens",
                    help="Path to validation dataset")
parser.add_argument("test_path", type=str, default="data/test.tokens",
                    help="Path to test dataset")
parser.add_argument("--model_name", type=str, default="distilgpt2", help="GPT model to use")
parser.add_argument("--model_output_dir", type=str, default="fine-tuning-output",
                    help="Output directory for fine-tuned model and checkpoints")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Fine-tuning epochs")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Fine-tuning learning rate")
parser.add_argument("--use_cuda", type=strtobool, default=True, help="Use GPU for training")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def fine_tune():
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "do_lower_case": True,
        "use_multiprocessing": False,
        "mlm": False,
        "num_train_epochs": args.num_train_epochs,
        'learning_rate': args.learning_rate,
        "max_seq_length": 300,
        "evaluate_during_training": True,
        "output_dir": args.model_output_dir
    }

    print(args.use_cuda)

    model = LanguageModelingModel('gpt2', args.model_name, use_cuda=args.use_cuda, args=train_args)

    model.eval_model(args.test_path)

    model.train_model(args.train_path,
                      eval_file=args.validation_path)

    model.eval_model(args.test_path)


def main():
    fine_tune()


if __name__ == '__main__':
    main()
