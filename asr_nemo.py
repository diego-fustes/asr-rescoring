import argparse
import json
import os

import nemo
import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.models import ASRConvCTCModel

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("wav_path", type=str, default="data/demo1.wav",
                    help="Path to wav file with the sound for transcription")

parser.add_argument("am_path", type=str, default="JasperNet10x5-En-Base.nemo",
                    help="Path to NeMo acoustic model")

parser.add_argument("lm_path", type=str, default="common_crawl_00.prune01111.trie.klm",
                    help="Path to KenLM N-Gram language model")

parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for acoustic model")

parser.add_argument("--beam_size", type=int, default=2048,
                    help="Beam size for beam search with language model")

parser.add_argument("--alpha", type=float, default=2.5,
                    help="Alpha parameter for beam search")

parser.add_argument("--beta", type=float, default=0.3,
                    help="Beta parameter for beam search")

parser.add_argument("--output_path", type=str, default="candidates.txt",
                    help="Path to KenLM N-Gram language model")

args = parser.parse_args()


def recognize_speech():
    if torch.cuda.is_available():
        neural_factory = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.GPU
            , optimization_level=nemo.core.Optimization.mxprO1
        )
    else:
        neural_factory = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU)

    # noinspection PyTypeChecker
    asr_model: ASRConvCTCModel = nemo_asr.models.ASRConvCTCModel.from_pretrained(model_info=args.am_path)
    asr_model.eval()

    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
        vocab=asr_model.vocabulary,
        beam_width=args.beam_size,
        alpha=args.alpha,
        beta=args.beta,
        lm_path=args.lm_path,
        num_cpus=max(os.cpu_count(), 1))

    # Create dummy manifest with single file
    manifest_path = "manifest.transcription"
    with open(manifest_path, 'w') as f:
        f.write(json.dumps({"audio_filepath": args.wav_path, "duration": 18000, "text": "todo"}))

    data_layer = nemo_asr.AudioToTextDataLayer(
        shuffle=False,
        manifest_filepath=manifest_path,
        labels=asr_model.vocabulary, batch_size=args.batch_size)

    audio_signal, audio_signal_len, _, _ = data_layer()

    log_probs, encoded_len = asr_model(input_signal=audio_signal, length=audio_signal_len)
    beam_predictions = beam_search_with_lm(
        log_probs=log_probs, log_probs_length=encoded_len)
    eval_tensors = [beam_predictions]

    tensors = neural_factory.infer(tensors=eval_tensors, use_cache=False, cache=False, offload_to_cpu=True)

    batch = tensors[-1][0]
    prediction = batch[0]
    candidates = [candidate[1] for candidate in prediction]

    with open(args.output_path, 'w') as f:
        for candidate in candidates:
            f.write(candidate + "\n")


def main():
    recognize_speech()


if __name__ == "__main__":
    main()
