import argparse

import numpy as np
import torch
from lm_scorer.models.gpt2 import GPT2LMScorer

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("predictions_path", type=str, default="predictions.txt",
                    help="Path to text file with candidate transcription predictions, one sentence per line")

parser.add_argument("--model_name", type=str, default="distilgpt2",
                    help="Rescoring model name or path to stored model")

parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for rescoring")

parser.add_argument("--output_path", type=str, default="best_prediction.txt",
                    help="Path to text file with best candidate prediction after rescoring")

args = parser.parse_args()


def rescore():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPT2LMScorer(
        model_name=args.model_name,
        device=device, batch_size=args.batch_size)

    with open(args.predictions_path) as f:
        candidates = f.readlines()

    scores = np.array(model.sentence_score(candidates, log=True))

    best_candidate = candidates[int(np.argmax(scores))]
    with open(args.output_path, 'w') as f:
        f.write(best_candidate)


def main():
    rescore()


if __name__ == "__main__":
    main()
