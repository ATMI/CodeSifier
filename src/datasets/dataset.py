import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from torch.utils import data


class Dataset(data.Dataset):
	def __init__(self, root_dir: Path, files: List[str], classes: Dict[str, int]):
		self.root_dir = root_dir
		self.files = files
		self.cls2idx = classes
		self.idx2cls = [0] * len(classes)
		for cls, idx in classes.items():
			self.idx2cls[idx] = cls

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file = self.files[idx]
		file = self.root_dir / file

		data = file.read_bytes()
		data = np.frombuffer(data, np.uint32)

		cls = self.cls2idx[file.parent.name]
		return data, cls

	@staticmethod
	def from_dir(dir_path: Path | str):
		if isinstance(dir_path, str):
			dir_path = Path(dir_path)

		files = []
		classes = {}

		for cls in dir_path.iterdir():
			if not cls.is_dir():
				continue
			classes[cls.name] = len(classes)

			for file in cls.iterdir():
				file = cls / file
				file = file.relative_to(dir_path)
				file = str(file)
				files.append(file)

		dataset = Dataset(dir_path, files, classes)
		return dataset


def main() -> None:
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command")

	prepare_parser = subparsers.add_parser("prepare")
	prepare_parser.add_argument("src", type=Path, help="Datasets folder path")
	prepare_parser.add_argument("dst", type=Path, help="Folder to save the prepared dataset")
	prepare_parser.add_argument("--proc", type=int, default=None, help="Number of processes to use")

	split_parser = subparsers.add_parser("split")
	split_parser.add_argument("src", type=Path, help="Datasets folder path")
	split_parser.add_argument("ratio", type=float, help="Train/test split ratio")
	split_parser.add_argument("--seed", type=int, default=None, help="Random seed")

	args = parser.parse_args()
	match args.command:
		case "prepare":
			# prepare_datasets(args.src, args.dst, args.proc)
			pass
		case "split":
			# split_datasets(args.src, args.ratio, args.seed)
			pass


if __name__ == "__main__":
	# main()
	tok = pd.read_csv(
		"/home/a/Projects/cimpl/cmake-build-release/tok.csv",
		delimiter="\t",
		encoding_errors="replace",
		on_bad_lines="skip",
		engine="pyarrow",
	)

	lang = pd.read_csv(
		"/home/a/Projects/cimpl/cmake-build-release/lang.csv",
		delimiter="\t",
		encoding="utf-8",
		engine="pyarrow",
		index_col=0,
	)

	frqs = tok.columns[4:].tolist()
	doc_frqs = frqs[0::2]
	term_frqs = frqs[1::2]

	lang_doc = lang.transpose()
	lang_doc = lang_doc.drop("term")
	lang_doc = lang_doc.rename(columns={name: f"{name} doc" for name in lang_doc.columns})
	lang_doc = lang_doc.reindex(columns=doc_frqs)

	r = np.any(tok[doc_frqs].values > 0.10 * lang_doc.values, axis=1)
	tok = tok[r]

	indices = set()
	for term in term_frqs:
		top = tok[term]
		top = top[top != 0]
		top = top.sort_values(ascending=False)

		cum_sum = top.cumsum()
		tot_sum = top.sum()

		ser = cum_sum < 0.975 * tot_sum
		top = top[ser].index

		indices |= set(top)

	indices = list(indices)
	tok = tok.loc[indices]


	def try_str(x):
		try:
			return str(x, encoding="utf-8")
		except Exception:
			return None


	tok["tok"] = tok["tok"].apply(try_str)
	tok = tok.dropna(subset=["tok"])

	tok = tok.sort_values(by="term", ascending=False)
	tok = tok[["type", "tok"]]
	tok = tok.reset_index(drop=True)
	tok.to_csv("/home/a/Projects/cimpl/cmake-build-release/vocab.csv", index_label="idx", sep="\t")
