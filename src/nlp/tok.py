import enum
from typing import Generator

import regex


class TokType(enum.IntEnum):
	LITERAL = 0
	NUMERIC = 1
	SPECIAL = 2
	NEWLINE = 3
	SPACE = 4

	def __repr__(self):
		return self.name

	def __str__(self):
		return self.__repr__()

	@classmethod
	def from_str(cls, s: str) -> 'TokType':
		return cls[s.upper()]


class Tok:
	type: TokType
	value: str

	def __init__(self, tok_type: TokType, tok_value: str):
		self.type = tok_type
		self.value = tok_value

	def __repr__(self):
		return f"{self.type} '{self.value}'"

	def __str__(self):
		return self.__repr__()

	def __hash__(self):
		return hash((self.type, self.value))

	def __eq__(self, other):
		if not isinstance(other, Tok):
			return NotImplemented
		return self.type == other.type and self.value == other.value


def tokenize(text: str) -> Generator[Tok, None, None]:
	pattern = regex.compile("(?<literal>\p{L}+)|(?<numeric>\d+)|(?<special>[^\p{L}\d\s])|(?<newline>\R+)|(?<space>\s+)")
	for match in pattern.finditer(text):
		for key, value in match.groupdict().items():
			if value is None:
				continue

			tok_type = TokType.from_str(key)
			tok_value = value.lower()
			tok = Tok(tok_type, tok_value)
			yield tok
