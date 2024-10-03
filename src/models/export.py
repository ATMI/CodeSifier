import argparse
from pathlib import Path

import torch

from model import Model


def main():
	args = argparse.ArgumentParser()
	args.add_argument("checkpoint", type=Path)

	args = args.parse_args()

	ckpt = torch.load(args.checkpoint)
	model_dict = ckpt["model"]
	vocab_len = ckpt["vocab_len"]
	embedding_len = ckpt["embedding_len"]
	class_num = ckpt["num_class"]

	model = Model(vocab_len, embedding_len, class_num)
	model.load_state_dict(model_dict)
	model.eval()

	with torch.no_grad():
		x = torch.randint(size=(16384,), low=0, high=vocab_len - 1)
		off = torch.tensor([0], dtype=torch.long)
		torch.onnx.export(
			model,
			args=(x, off),
			f="model.onnx",
			input_names=["tokens", "offsets"],
			output_names=["probs"],
			dynamic_axes={
				"tokens": {
					0: "width"
				}
			}
		)

	idx2cls = ckpt["idx2cls"]
	with open("idx2cls.txt", "w") as f:
		f.writelines("\n".join(idx2cls))


if __name__ == "__main__":
	main()
