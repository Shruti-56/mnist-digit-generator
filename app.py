import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- Generator Model ----
class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], 1)
        x = self.model(x)
        return x.view(-1, 1, 28, 28)

# ---- Load model ----
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

generator = load_model()

# ---- UI ----
st.title("🧠 MNIST Digit Generator (METI Internship)")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = torch.zeros(5, 10)
    labels[range(5), [digit]*5] = 1

    with torch.no_grad():
        images = generator(z, labels).squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
