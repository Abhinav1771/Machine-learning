import streamlit as st
import torch  
import torch.nn as nn 
import torch.optim as optim
from torch.distributions import Categorical

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

# Initialize the model
vocab_size = len(stoi)
model = NextWordMLP(vocab_size, embedding_dim, block_size, hidden_size, activation_func)
model.to(device)

# Define optimizer and loss function
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

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
if st.sidebar.checkbox("Show Embedding Visualization"):
    # Code for visualizing embeddings can go here
    st.write("Embedding visualization (if enabled)")

