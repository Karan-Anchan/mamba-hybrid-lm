"""Turn a dataset into flat uint16 token files (the nanoGPT trick).

Encode every doc, drop an <|endoftext|> after it, and stream the ids straight to disk as uint16.
uint16 works because our vocab (16k) is < 65536, and it halves the file size vs int32. Writing as I
go means memory stays flat no matter how big the corpus is.

Ends up in data/<dataset>/:
    train.bin, val.bin   token streams (read back with np.memmap)
    meta.json            vocab size, dtype, eot id, token counts

    python -m src.data.prepare_data --dataset tinystories --train-docs 50000 --val-docs 22000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from src.data.train_tokenizer import EOT, text_iterator


def _stream_split(ds_iter, tok: Tokenizer, eot_id: int, out_path: Path) -> int:
    # encode -> append eot -> write bytes, one doc at a time. returns total tokens written
    n_tokens = 0
    with open(out_path, "wb") as f:
        for text in ds_iter:
            ids = tok.encode(text).ids
            ids.append(eot_id)
            arr = np.asarray(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            n_tokens += arr.size
    return n_tokens


def _val_iterator(dataset: str, docs: int):
    # tinystories ships a real validation split; owt doesn't, so I just take a head slice for val
    from datasets import load_dataset

    if dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        for i, ex in enumerate(ds):
            if i >= docs:
                break
            yield ex["text"]
    else:
        for text in text_iterator(dataset, docs):
            yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "openwebtext"])
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--train-docs", type=int, default=50_000)
    ap.add_argument("--val-docs", type=int, default=22_000)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id(EOT)
    out_dir = Path("data") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # do val first (it's small) so a mistake shows up fast before the long train pass
    print(f"[val]   encoding <= {args.val_docs:,} docs ...")
    n_val = _stream_split(_val_iterator(args.dataset, args.val_docs), tok, eot_id, out_dir / "val.bin")
    # for datasets without a real val split I carved val from the head, so train must skip past it
    # (otherwise the same docs land in both — a leak). TinyStories has its own val split, so skip=0.
    train_skip = 0 if args.dataset == "tinystories" else args.val_docs
    print(f"[train] encoding <= {args.train_docs:,} docs (skip {train_skip}) ...")
    n_train = _stream_split(text_iterator(args.dataset, args.train_docs, skip=train_skip),
                            tok, eot_id, out_dir / "train.bin")

    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "eot_id": eot_id,
        "train_tokens": n_train,
        "val_tokens": n_val,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done. train={n_train:,} tok  val={n_val:,} tok  -> {out_dir}/  (meta.json written)")


if __name__ == "__main__":
    main()
