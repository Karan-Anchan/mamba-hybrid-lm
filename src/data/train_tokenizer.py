"""Train a 16k byte-level BPE tokenizer.

Small custom vocab (16k) on purpose: with a 50M budget a big GPT-2-size embedding would eat most of
my params, and I'd rather spend them on the mixing layers (see D-ARCH-01). Byte-level BPE means no
UNK and it copes with any input. One special token, <|endoftext|>, marks doc boundaries.

    python -m src.data.train_tokenizer --dataset tinystories --docs 200000 --vocab 16000
    # saves data/tokenizer/tokenizer.json (that path is gitignored; the script is what's tracked)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


def text_iterator(dataset: str, docs: int, skip: int = 0):
    # streaming so I don't pull the whole dataset down; take `docs` examples after skipping `skip`.
    # skip is how I carve a non-overlapping val/train split out of a dataset that has no val split.
    from datasets import load_dataset

    if dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        key = "text"
    elif dataset == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        key = "text"
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if i >= skip + docs:
            break
        yield ex[key]


def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    # GPT-2 style: byte-level BPE, no UNK token
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # seed all 256 bytes so nothing is OOV
        show_progress=True,
    )
    return tok, trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "openwebtext"])
    ap.add_argument("--docs", type=int, default=200_000, help="how many docs to train on")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--out", default="data/tokenizer/tokenizer.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, trainer = build_tokenizer(args.vocab)
    print(f"training {args.vocab}-vocab byte-level BPE on <= {args.docs:,} {args.dataset} docs ...")
    tok.train_from_iterator(text_iterator(args.dataset, args.docs), trainer=trainer)
    tok.save(str(out))

    eot_id = tok.token_to_id(EOT)
    print(f"done. vocab={tok.get_vocab_size()}  <|endoftext|> id={eot_id}  saved -> {out}")


if __name__ == "__main__":
    main()
