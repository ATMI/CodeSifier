from vocab import Vocab, TokType, Tok
import regex


class Tokenizer:
	def __init__(self, vocab: Vocab):
		self.regex = regex.compile("(\\p{L}+)|(\\d+)|(\\s+)|(.)")
		self.vocab = vocab
		self.types = [
			TokType.LITERAL,
			TokType.NUMERIC,
			TokType.BLANCHE,
			TokType.SPECIAL,
		]

	def tokenize(self, s: str):
		for match in self.regex.finditer(s):
			for idx, value in enumerate(match.groups()):
				if value is None:
					continue

				typ = self.types[idx]
				tok = Tok(typ, value)
				tok = self.vocab[tok]
				if tok is not None:
					yield tok
