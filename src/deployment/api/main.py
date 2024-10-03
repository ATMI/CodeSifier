import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from tokenizer import Tokenizer
from vocab import Vocab

vocab = Vocab.from_file("vocab.csv")
tokenizer = Tokenizer(vocab)
ort_session = ort.InferenceSession("model.onnx")

with open("idx2cls.txt", "r") as f:
	idx2cls = f.readlines()
idx2cls = [s.strip() for s in idx2cls]

app = FastAPI()


@app.get("/")
async def root():
	return RedirectResponse(url="/docs", status_code=302)


class Input(BaseModel):
	code: str


@app.post("/classify/")
async def predict(inp: Input):
	toks = tokenizer.tokenize(inp.code)
	toks = list(toks)
	offsets = [0]

	res = ort_session.run(
		["probs"],
		{
			"tokens": toks,
			"offsets": offsets,
		}
	)

	res = res[0]
	res = np.argmax(res)
	res = idx2cls[res]
	return res
