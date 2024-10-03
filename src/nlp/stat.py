import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict

from src.nlp.tok import tokenize, Tok


def count_tokens_in_file(file_path: Path) -> Optional[Counter[Tok]]:
	try:
		with file_path.open("r") as f:
			text = f.read()

		tokens = tokenize(text)
		counter = Counter(tokens)
	except Exception as e:
		print(file_path, e)
		counter = None
	return counter


def __count_tokens_in_file__(pair: Tuple[Path, str]) -> Optional[Tuple[str, Counter[Tok]]]:
	file_path, language = pair
	counter = count_tokens_in_file(file_path)
	if counter is None:
		return None
	return language, counter


def count_tokens(
	files: Iterable[Tuple[Path, str]],
	proc: Optional[int] = None,
	chunk: Optional[int] = None,
) -> Tuple[Dict[str, Counter[Tok]], Dict[str, Counter[Tok]], Counter[str]]:
	if proc is None:
		proc = max(1, mp.cpu_count() // 2)

	if chunk is None:
		chunk = 5 * proc

	tok_lang = defaultdict(Counter)
	tok_lang_doc = defaultdict(Counter)
	doc_lang = Counter()

	with mp.Pool(proc) as p:
		results = p.imap_unordered(__count_tokens_in_file__, files, chunksize=chunk)
		for result in results:
			if result is None:
				continue

			language, counter = result
			tokens = counter.keys()

			tok_lang[language].update(counter)
			tok_lang_doc[language].update(tokens)
			doc_lang.update([language])

	return tok_lang, tok_lang_doc, doc_lang
