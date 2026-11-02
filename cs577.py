# %% [markdown]
# # Assignment 2 - Recurrent Neural Networks
# 
# 

# %% [markdown]
# ## Programming (Full points: 100)

# %% [markdown]
# In this assignment, our goal is to use PyTorch to implement Recurrent Neural Networks (RNN) for sentiment analysis task. Sentiment analysis is to classify sentences (input) into certain sentiments (output labels), which includes positive, negative and neutral.
# 
# We will use a benckmark dataset, SST, for this assignment.
# * we download the SST dataset from torchtext package, and do some preprocessing to build vocabulary and split the dataset into training/validation/test sets. You don't need to modify the code in this step.
# 

# %%
import os
os.chdir("/workspaces/cs577-homework")
print(os.getcwd())

# %%
import copy
import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data
from torchtext import datasets

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.LabelField()

# load data splits
train_data, val_data, test_data = datasets.SST.splits(TEXT, LABEL)

# build dictionary
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

# hyperparameters
vocab_size = len(TEXT.vocab)
label_size = len(LABEL.vocab)
padding_idx = TEXT.vocab.stoi['<pad>']
embedding_dim = 128
hidden_dim = 128

# build iterators
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data), 
    batch_size=32)

# %% [markdown]
# * define the training and evaluation function in the cell below.
# ### (25 points)
# 

# %%
# your code here
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        acc = (predictions.argmax(dim=1) == batch.label).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = (predictions.argmax(dim=1) == batch.label).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# %% [markdown]
# * build a RNN model for sentiment analysis in the cell below.
# We have provided several hyperparameters we needed for building the model, including vocabulary size (vocab_size), the word embedding dimension (embedding_dim), the hidden layer dimension (hidden_dim), the number of layers (num_layers) and the number of sentence labels (label_size). Please fill in the missing codes, and implement a RNN model.
# ### (40 points)

# %%
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, padding_idx):
        super(RNNClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.num_layers = 1

        # add the layers required for sentiment analysis.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)

        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.label_size)

    def zero_state(self, batch_size):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.embedding.weight.device
        )

    def forward(self, text):
        # implement the forward function of the model.
        embedding = self.embedding(text)

        batch_size = text.size(0)
        hidden = self.zero_state(batch_size)

        output, hidden = self.rnn(embedding, hidden)

        predictions = self.fc(hidden[-1])

        return predictions

# %% [markdown]
# * train the model and compute the accuracy in the cell below.
# ### (20 points)

# %%
model = RNNClassifier(
    vocab_size,
    embedding_dim,
    hidden_dim,
    label_size,
    padding_idx
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    train_loss, train_acc = train(
        model, train_iter, optimizer, criterion
    )

    val_loss, val_acc = evaluate(
        model, val_iter, criterion
    )

    print(
        f"Epoch {epoch+1}/20 | "
        f"Train Accuracy: {train_acc:.2%} | "
        f"Validation Accuracy: {val_acc:.2%}"
    )

test_loss, test_acc = evaluate(
    model, test_iter, criterion
)

print(f"RNN Test Accuracy: {test_acc:.2%}")

# %% [markdown]
# * try to train a model with better accuracy in the cell below. For example, you can use different optimizers such as SGD and Adam. You can also compare different hyperparameters and model size.
# ### (15 points), to obtain FULL point in this problem, the accuracy needs to be higher than 70%

# %%
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=32
)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 label_size, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, label_size)
        self.padding_idx = padding_idx

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        lengths = (text != self.padding_idx).sum(dim=1).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        return self.fc(self.dropout(hidden[-1]))


# Create model
lstm_model = LSTMClassifier(
    vocab_size,
    embedding_dim,
    128,
    label_size,
    padding_idx
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    lstm_model.parameters(),
    lr=0.001,
    weight_decay=0.001
)

# Training settings
best_val_acc = 0
best_model_state = None
patience = 5
no_improvement = 0

# Train model
for epoch in range(20):
    train_loss, train_acc = train(
        lstm_model, train_iter, optimizer, criterion
    )

    val_loss, val_acc = evaluate(
        lstm_model, val_iter, criterion
    )

    print(
        f"Epoch {epoch+1}/20 | "
        f"Train Accuracy: {train_acc:.2%} | "
        f"Validation Accuracy: {val_acc:.2%}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(
            lstm_model.state_dict()
        )
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print("Early stopping triggered.")
        break

# Restore best model
lstm_model.load_state_dict(best_model_state)

# Evaluate on test set
test_loss, test_acc = evaluate(
    lstm_model, test_iter, criterion
)

print(f"\nBest Validation Accuracy: {best_val_acc:.2%}")
print(f"LSTM Test Accuracy: {test_acc:.2%}")


