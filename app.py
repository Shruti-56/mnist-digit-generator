import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- 1) Generator definition ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.main(x).view(-1, 1, 28, 28)

nz = 100
num_classes = 10
G = Generator()
G.load_state_dict(torch.load("dcgan_generator.pt", map_location="cpu"))
G.eval()

# --- 3) Streamlit UI ---
st.title("🧑‍🎨 Handwritten Digit Generator (DCGAN)")

digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate 5 Images"):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax in axes:
        noise = torch.randn(1, nz)
        label = torch.zeros(1, num_classes)
        label[0, digit] = 1
        fake_img = G(noise, label).detach().numpy().reshape(28, 28)
        ax.imshow(fake_img, cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
