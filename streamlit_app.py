import streamlit as st
import torch  
import torch.nn as nn 
import torch.optim as optim
from torch.distributions import Categorical
import requests
import re
import os
import torch.optim as optim
from pprint import pprint

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'






# Function to initialize the model
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, hidden_size, activation_func):
        super(NextWordMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_size)
        self.activation = activation_func
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x





# Set up Streamlit UI
st.title("Next Word Prediction")
st.sidebar.header("Model Parameters")

# Model parameters with default values
embedding_dim = st.sidebar.slider("Embedding Dimension", 32, 128, 64)
block_size = st.sidebar.slider("Context Length", 1, 10, 5)
hidden_size = st.sidebar.slider("Hidden Layer Size", 512, 2048, 1024)
activation_choice = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "LeakyReLU"])
random_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=10000, value=42)
top_k = st.sidebar.slider("Top K Predictions", 1, 10, 5)

# Map activation functions to PyTorch functions
activation_dict = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "LeakyReLU": nn.LeakyReLU()}
activation_func = activation_dict[activation_choice]

# Set random seed for reproducibility
torch.manual_seed(random_seed)







# Load the vocabulary
# Ensure that `stoi` and `itos` mappings are created here for vocabulary lookup
# Download the dataset
url = 'https://www.gutenberg.org/files/1661/1661-0.txt'
response = requests.get(url)
text = response.text

# Preprocess text: remove special characters, convert to lowercase, etc.
text = re.sub('[^a-zA-Z0-9 \.]', '', text).lower()

# Split text into words
words = text.split()
# print(f"Sample words: {words[:10]}")

# Filter out very short or very long words for a consistent vocabulary
words = [word for word in words if 2 < len(word) < 10]

# Remove words having non alphabets
words = [word for word in words if word.isalpha()]

# Create vocabulary and mappings
unique_words = sorted(set(words))
vocab_size = len(unique_words)
stoi = {word: i for i, word in enumerate(unique_words)}
itos = {i: word for word, i in stoi.items()}

# print(f"Vocabulary size: {vocab_size}")
# pprint(itos)


# # Convert the text into a sequence of integer indices
# data = [stoi[word] for word in words]
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Create dataset for next-word prediction
# def create_dataset(data, block_size):
#     X, Y = [], []
#     for i in range(len(data) - block_size):
#         context = data[i:i + block_size]
#         ix = data[i + block_size]

#         # Print the context and next word in the desired format
#         #print(' '.join(itos[j] for j in context), '--->', itos[ix])

#         X.append(context)
#         Y.append(ix)

#     return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# # Generate X and Y with printed contexts and next words
# X, Y = create_dataset(data, block_size)
# # Split the data into training and validation sets
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the model
vocab_size = len(stoi)
model = NextWordMLP(vocab_size, embedding_dim, block_size, hidden_size, activation_func)
model.to(device)

# Define optimizer and loss function
epochs = 10
batch_size=64
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
# # Training loop
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     for i in range(0, len(X_train), batch_size):
#         x_batch = X_train[i:i + batch_size].to(device)
#         y_batch = Y_train[i:i + batch_size].to(device)

#         # Forward pass
#         y_pred = model(x_batch)

#         # Compute loss
#         loss = criterion(y_pred, y_batch)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     # Validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for i in range(0, len(X_val), batch_size):
#             x_val_batch = X_val[i:i + batch_size].to(device)
#             y_val_batch = Y_val[i:i + batch_size].to(device)

#             # Forward pass
#             y_val_pred = model(x_val_batch)

#             # Compute loss
#             loss = criterion(y_val_pred, y_val_batch)
#             val_loss += loss.item()

#     #print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(X_train):.4f}, Validation Loss: {val_loss / len(X_val):.4f}")


# Function to predict next words
def predict_next_words(model, context, top_k):
    model.eval()
    context_indices = torch.tensor([stoi.get(word, 0) for word in context], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(context_indices)
    probs = torch.softmax(logits, dim=-1).squeeze()
    top_k_indices = torch.topk(probs, top_k).indices
    return [itos[idx.item()] for idx in top_k_indices]

# User input for the context
context = st.text_input("Enter the context text", "the adventure of sherlock holmes").lower().split()
if st.button("Predict"):
    next_words = predict_next_words(model, context, top_k)
    st.write(f"Top {top_k} predictions: {', '.join(next_words)}")

# Embedding visualization (optional)
# if st.sidebar.checkbox("Show Embedding Visualization"):
#     # Code for visualizing embeddings can go here
#     st.write("Embedding visualization (if enabled)")

