from torch import nn


class Model(nn.Module):
	def __init__(self, vocab_len: int, embedding_len: int, class_num: int):
		super(Model, self).__init__()

		self.embedding = nn.EmbeddingBag(vocab_len, embedding_len)
		self.fc = nn.Linear(embedding_len, class_num)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x, off):
		x = self.embedding(x, off)
		x = self.fc(x)
		x = self.softmax(x)
		return x
