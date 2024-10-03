from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm

from src.datasets.dataset import Dataset
from src.deployment.api.vocab import Vocab
from src.models.model import Model


def calculate_conf(y_pred, y_true):
	predictions = torch.nn.functional.one_hot(y_pred, num_classes=27)  # TODO: AAAAA
	targets = torch.nn.functional.one_hot(y_true, num_classes=27)  # TODO: AAAAA

	tp = torch.sum(predictions * targets, dim=0)
	tn = torch.sum((1 - predictions) * (1 - targets), dim=0)
	fp = torch.sum((1 - targets) * predictions, dim=0)
	fn = torch.sum(targets * (1 - predictions), dim=0)

	return tp, tn, fp, fn


def calculate_metrics(tp, fp, fn):
	eps = 1e-7
	precision = tp / (tp + fp + eps)
	recall = tp / (tp + fn + eps)
	f1_score = 2 * (precision * recall) / (precision + recall + eps)

	macro_precision = torch.mean(precision)
	macro_recall = torch.mean(recall)
	macro_f1_score = torch.mean(f1_score)

	return macro_precision.item(), macro_recall.item(), macro_f1_score.item()


def train_epoch(
	model: nn.Module,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	train_loader: data.DataLoader,
	device: torch.device,
	epoch: int,
):
	model.train()
	model.to(device)

	running_loss = 0.0
	running_scores = None
	bar = tqdm(train_loader, desc=f"Epoch: {epoch}", position=0, leave=True)

	for batch_idx, (x, off, y) in enumerate(bar):
		x, off, y = x.to(device), off.to(device), y.to(device)
		optimizer.zero_grad()
		y_pred = model(x, off)

		loss = criterion(y_pred, y)
		running_loss += loss.item()

		_, y_pred = torch.max(y_pred, axis=1)
		scores = calculate_conf(y_pred, y)

		running_scores = running_scores and tuple(map(sum, zip(running_scores, scores))) or scores
		tp, tn, fp, fn = running_scores
		pre, rec, f1 = calculate_metrics(tp, fp, fn)

		loss.backward()
		optimizer.step()

		bar.set_postfix(loss=running_loss / (batch_idx + 1), pre=pre, rec=rec, f1=f1)

	return running_loss / len(train_loader)


def test_epoch(
	model: nn.Module,
	criterion: nn.Module,
	test_loader: data.DataLoader,
	device: torch.device,
	epoch: int,
):
	with torch.no_grad():
		model.eval()
		model.to(device)

		running_loss = 0.0
		running_scores = None
		bar = tqdm(test_loader, desc=f"Epoch: {epoch}", position=0, leave=True)

		for batch_idx, (x, off, y) in enumerate(bar):
			x, off, y = x.to(device), off.to(device), y.to(device)
			y_pred = model(x, off)

			loss = criterion(y_pred, y)
			running_loss += loss.item()

			_, y_pred = torch.max(y_pred, axis=1)
			scores = calculate_conf(y_pred, y)

			running_scores = running_scores and tuple(map(sum, zip(running_scores, scores))) or scores
			tp, tn, fp, fn = running_scores
			pre, rec, f1 = calculate_metrics(tp, fp, fn)

			bar.set_postfix(loss=running_loss / (batch_idx + 1), f1=f1, pre=pre, rec=rec)

		return running_loss / len(test_loader)


if __name__ == "__main__":
	vocab = Vocab.from_file("/home/a/Projects/CodeSifier/data/processed/vocab.csv")
	dataset = Dataset.from_dir("/home/a/Projects/CodeSifier/data/processed")

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_set, test_set = data.random_split(dataset, [train_size, test_size])


	def collate_fn(batch):
		x, y = zip(*batch)

		x = [torch.from_numpy(t.copy()).long() for t in x]
		sizes = torch.tensor([0] + [t.size(0) for t in x][:-1])

		off = torch.cumsum(sizes, dim=0)
		x = torch.cat(x)
		y = torch.tensor(y)

		return x, off, y


	train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
	test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

	vocab_len = len(vocab)
	embedding_len = 128
	class_num = len(dataset.cls2idx)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model(vocab_len, embedding_len, class_num)
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	checkpoint_dir = Path("checkpoints")
	checkpoint_dir.mkdir(exist_ok=True)

	for epoch in tqdm(range(20)):
		train_epoch(model, criterion, optimizer, train_loader, device, epoch)
		test_epoch(model, criterion, test_loader, device, epoch)

		ckpt = {
			"model": model.state_dict(),
			"vocab_len": vocab_len,
			"embedding_len": embedding_len,
			"num_class": class_num,
			"idx2cls": dataset.idx2cls,
			"optim": optimizer.state_dict(),
			"epoch": epoch,
		}
		ckpt_path = checkpoint_dir / f"{epoch}.pt"
		torch.save(ckpt, ckpt_path)
