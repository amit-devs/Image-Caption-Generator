# Image Caption Generator Using CNN, Attention Mechanism and LSTM

An AI-based web application that automatically generates descriptive captions for uploaded images using a deep learning architecture combining a **CNN Encoder, Attention Mechanism, and LSTM Decoder**.

The project uses **ResNet50** as the CNN encoder to extract visual features from an image. An attention mechanism helps the model focus on relevant regions of the image while generating each word, and an LSTM decoder generates the final caption sequentially.

---

## Features

- Upload an image through a modern web interface
- Generate automatic image captions
- CNN-based visual feature extraction using ResNet50
- Attention mechanism for focusing on important image regions
- LSTM-based sequential caption generation
- Image preview before processing
- Drag-and-drop image upload
- Caption regeneration interface
- Caption improvement interface
- Responsive and animated user interface
- Flask-based web application

---

## System Architecture

```text
                 Input Image
                     │
                     ▼
              ┌─────────────┐
              │  ResNet50   │
              │ CNN Encoder │
              └──────┬──────┘
                     │
                     ▼
          Spatial Image Features
                     │
                     ▼
          ┌────────────────────┐
          │ Attention Mechanism│
          └─────────┬──────────┘
                    │
                    ▼
             ┌────────────┐
             │    LSTM    │
             │   Decoder  │
             └─────┬──────┘
                   │
                   ▼
            Generated Caption
```

---

## Technologies Used

### Deep Learning

- Python
- PyTorch
- Torchvision
- ResNet50
- LSTM
- Attention Mechanism

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Dataset and Processing

- Flickr8k Dataset
- Pillow
- NumPy
- Pandas
- NLTK

### Development Tools

- PyCharm
- VS Code
- Git
- GitHub
- Python Virtual Environment

---

## Dataset

The project uses the **Flickr8k dataset**, which contains images paired with multiple natural-language captions.

The dataset is used for vocabulary creation and model training.

The dataset is not included in this repository because of its size.

Expected dataset structure:

```text
dataset/
├── Images/
└── captions.txt
```

---

## How the System Works

### 1. Image Upload

The user uploads an image through the Flask-based web interface.

### 2. Image Preprocessing

The uploaded image is resized to `224 × 224`, converted into a tensor, and normalized using ImageNet normalization values.

### 3. CNN Feature Extraction

A pretrained **ResNet50** network is used as the CNN encoder to extract visual features from the uploaded image.

The final classification layers are removed so that spatial feature information can be preserved.

### 4. Attention Mechanism

The attention mechanism calculates the importance of different spatial regions of the image while generating each word.

This allows the decoder to focus on relevant visual features during caption generation.

### 5. LSTM Decoder

The LSTM decoder receives the attended image features along with word embeddings and generates the caption sequentially.

### 6. Caption Generation

The model predicts one word at a time until the `<END>` token is generated or the maximum caption length is reached.

---

## Project Structure

```text
Image Caption Gen/
│
├── app.py
├── model.py
├── vocab.py
├── dataset.py
├── train.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── encoder.pth
│   ├── decoder.pth
│   └── vocab.pkl
│
├── dataset/
│   ├── Images/
│   └── captions.txt
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── uploads/
│
└── templates/
    └── index.html
```

> Note: The `dataset/`, trained model files, and uploaded images are excluded from the Git repository using `.gitignore`.

---

## Model Components

| Component | Purpose |
|---|---|
| ResNet50 | Extracts visual features from images |
| Linear Projection | Converts CNN features into the encoder dimension |
| Attention Mechanism | Focuses on relevant image regions |
| Word Embedding | Converts words into numerical vectors |
| LSTM Decoder | Generates captions sequentially |
| Fully Connected Layer | Predicts the next word |

---

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd "Image Caption Gen"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Place the Flickr8k dataset inside the `dataset` folder.

The structure should be:

```text
dataset/
├── Images/
└── captions.txt
```

### 6. Train the Model

```bash
python train.py
```

The trained model files are generated inside the `models/` directory.

### 7. Run the Web Application

```bash
python app.py
```

Open the local Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

---

## Future Improvements

- Improve caption quality with more extensive model training
- Implement beam search for better caption generation
- Fine-tune the CNN encoder
- Add BLEU, METEOR and CIDEr evaluation metrics
- Improve caption regeneration
- Enhance caption refinement functionality
- Support larger image-caption datasets
- Deploy the application as a cloud-based web service

---

## Limitations

- Caption quality depends on the training data and model training.
- The model may not correctly identify every object or scene.
- Complex images can produce less accurate descriptions.
- The generated vocabulary is limited to the words learned from the training dataset.

---

## Project Team

This project was developed collaboratively by:

1. **Amit** — Machine Learning & Backend Development
2. **Mehul** — Frontend & UI Development
3. **Vivek** — Dataset, Testing & Documentation

---

## Project Type

**Academic Deep Learning Project**

### Core Architecture

**CNN + Attention Mechanism + LSTM**

---

## License

This project is developed for educational and academic purposes.
