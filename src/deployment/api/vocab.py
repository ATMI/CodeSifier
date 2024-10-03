import csv
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict

import numpy as np


class TokType(Enum):
	LITERAL = auto()
	NUMERIC = auto()
	BLANCHE = auto()
	SPECIAL = auto()

	@classmethod
	def from_str(cls, s: str) -> 'TokType':
		try:
			return cls[s.upper()]
		except KeyError:
			raise ValueError(f"Invalid token type: {s}")


@dataclass(frozen=True)
class Tok:
	type: TokType
	value: str

	def __hash__(self):
		match self.type:
			case TokType.LITERAL | TokType.SPECIAL:
				return hash(self.value)
			case TokType.BLANCHE | TokType.NUMERIC:
				return hash(self.type)

	def __eq__(self, other):
		if not isinstance(other, Tok):
			return NotImplemented

		match self.type:
			case TokType.LITERAL | TokType.SPECIAL:
				return self.value == other.value
			case TokType.BLANCHE | TokType.NUMERIC:
				return self.type == other.type


class Vocab:
	def __init__(self, tok2idx: Dict[Tok, int]):
		self.tok2idx = tok2idx
		self.idx2tok = [0] * len(tok2idx)

		for tok, idx in self.tok2idx.items():
			self.idx2tok[idx] = tok

	def __len__(self):
		return len(self.tok2idx) + 2

	def __getitem__(self, idx: Tok | int) -> Tok | int | None:
		if isinstance(idx, (int, np.integer)):
			if idx < 0 or idx >= len(self.idx2tok):
				return None
			return self.idx2tok[idx]
		elif isinstance(idx, Tok):
			return self.tok2idx.get(idx)

	def unk(self):
		return len(self.tok2idx)

	@staticmethod
	def from_file(vocab_file: Path | str):
		if isinstance(vocab_file, str):
			vocab_file = Path(vocab_file)

		tok2idx = {}

		with open(vocab_file, "r") as f:
			reader = csv.DictReader(f, delimiter="\t")
			for row in reader:
				tok = row["tok"]
				idx = row["idx"]

				typ = row["type"]
				typ = TokType.from_str(typ)

				tok = Tok(typ, tok)
				tok2idx[tok] = int(idx)

		vocab = Vocab(tok2idx)
		return vocab
