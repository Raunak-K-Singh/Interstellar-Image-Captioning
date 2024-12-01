import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import json
from PIL import Image as PILImage
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
from models import ImageCaptioningModel


class InterstellarCaptioningApp:
    def __init__(self, master):
        # Set up the main window with a cosmic theme
        self.master = master
        master.title("Cosmic Image Decoder")

        master.configure(bg='#0C1445')  # Deep space blue background
        master.geometry('600x800')  # Larger window

        # Main title label
        title_label = tk.Label(master, text="🚀 Interstellar Image Decoder", font=("Helvetica", 16), bg='#0C1445',
                               fg='#00FFFF')
        title_label.pack(pady=20)

        # Image preview area
        self.image_preview = tk.Label(master, text="Upload a Cosmic Snapshot", bg='#1A2347', fg='#FFFFFF', width=50,
                                      height=20)
        self.image_preview.pack(pady=10)

        button_frame = tk.Frame(master, bg='#0C1445')
        button_frame.pack(pady=10)

        # Upload button
        upload_button = tk.Button(button_frame, text="🛰️ Scan Cosmic Image", command=self.upload_image,
                                  font=("Russo One", 12), bg='#2196F3', fg='white', activebackground='#1976D2')
        upload_button.pack(side=tk.LEFT, padx=10)

        # Generate Caption button
        caption_button = tk.Button(button_frame, text="📡 Decode Transmission", command=self.generate_caption,
                                   font=("Russo One", 12), bg='#4CAF50', fg='white', activebackground='#388E3C')
        caption_button.pack(side=tk.LEFT, padx=10)

        # Output area
        self.text_output = tk.Text(master, height=6, width=70, font=("Courier", 12), bg='#121212', fg='#00FF00',
                                   insertbackground='white')
        self.text_output.pack(pady=20)

        # Initialize model and other attributes
        self.model, self.vocab_dict = self.load_model()
        self.current_image = None

    def upload_image(self):
        """Upload and preview a cosmic image"""
        file_path = filedialog.askopenfilename(filetypes=[("Cosmic Images", "*.jpg;*.jpeg;*.png;*.gif")])

        if file_path:
            try:
                img = PILImage.open(file_path).convert("RGB")
                img.show()  # Display the uploaded image

                # Store the uploaded image path for later use.
                self.current_image = file_path

                messagebox.showinfo("Cosmic Scan", "Image successfully captured!")

            except Exception as e:
                messagebox.showerror("Transmission Error", f"Could not process image: {e}")

    def generate_caption(self):
        """Generate a cosmic narrative for the image"""
        if not self.current_image:
            messagebox.showwarning("Transmission Failed", "No cosmic image detected. Scan an image first!")
            return

        try:
            image_tensor = transform_image(self.current_image)

            with torch.no_grad():
                output = self.model(image_tensor.unsqueeze(0).to(torch.device('cpu')))

                # Placeholder for actual caption generation logic
                generated_caption = "A mysterious cosmic landscape revealing secrets of the universe"

                concepts = ["universe", "cosmic", "landscape"]
                graph_plt = self.create_cosmic_knowledge_graph(concepts)

                plt.show()  # Show knowledge graph plot.

                # Update output with caption and additional info.
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(tk.END, f"🛰️ Decoded Transmission:\n{generated_caption}\n\n")
                self.text_output.insert(tk.END, f"🌠 Key Cosmic Concepts: {', '.join(concepts)}")

        except Exception as e:
            messagebox.showerror("Decoding Error", f"Cosmic transmission failed: {e}")

    def load_model(self):
        vocab_path = 'data/captions.json'

        with open(vocab_path) as f:
            vocab_data = json.load(f)

        vocab_size = len(vocab_data['vocab'])

        model = ImageCaptioningModel(vocab_size=vocab_size)

        model_path = 'models/model.pth'

        model.load_state_dict(torch.load(model_path))

        model.eval()

        return model, vocab_data['vocab']

    def create_cosmic_knowledge_graph(self, concepts):
        G = nx.Graph()

        for concept in concepts:
            G.add_node(concept)

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                G.add_edge(concepts[i], concepts[j])

        plt.figure(figsize=(10, 6), facecolor='#121212')

        pos = nx.spring_layout(G, k=0.5, iterations=50)

        nx.draw(G, pos, with_labels=True,
                node_color='#4CAF50',
                edge_color='#2196F3',
                node_size=1500,
                font_size=10,
                font_weight='bold',
                font_color='white')

        plt.title("Cosmic Concept Mapping", color='white', fontsize=15)
        plt.tight_layout()

        return plt


def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = PILImage.open(image_path).convert("RGB")
    return transform(image)


if __name__ == "__main__":
    root = tk.Tk()
    app = InterstellarCaptioningApp(root)
    root.mainloop()
