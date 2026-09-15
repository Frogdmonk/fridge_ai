# Fridge AI

**Turn a short refrigerator or kitchen video into a recipe using the ingredients actually detected in the video.**

Fridge AI is an AI-powered computer vision and retrieval-augmented recipe generation system. It analyzes uploaded videos, detects available ingredients, retrieves relevant recipes using semantic search, and generates a recipe grounded in the detected ingredients.

##  Features

* **Video-based ingredient detection**
* <img width="494" height="318" alt="image" src="https://github.com/user-attachments/assets/8b34a80c-50da-4ad3-acfc-38c1787c6ee2" />

*  **YOLOv8 object detection**
* <img width="791" height="386" alt="image" src="https://github.com/user-attachments/assets/1d1ea6f3-4365-48d3-8614-4eff4f56822d" />

*  **Motion-based keyframe selection with OpenCV**
*  **Annotated ingredient detection**
*  **Semantic recipe retrieval**
* <img width="455" height="140" alt="image" src="https://github.com/user-attachments/assets/783be371-d82b-4d1d-a61a-b45d0cf63a35" />

*  **Ingredient-overlap ranking**
* <img width="936" height="323" alt="image" src="https://github.com/user-attachments/assets/eb6a0798-3316-4c92-a98e-6d7cb2032bb5" />

*  **LLM-powered recipe generation**
* <img width="920" height="359" alt="image" src="https://github.com/user-attachments/assets/f6aff04f-283e-4595-88fb-c40a07db37b2" />
<img width="933" height="314" alt="image" src="https://github.com/user-attachments/assets/ca8548bd-9db3-4a0f-a0dd-4f18f4fa2450" />

*  **Flask web application**

## AI Pipeline

```text
                  Upload Video
                       │
                       ▼
              ┌─────────────────┐
              │     OpenCV      │
              │ Video Processing│
              └────────┬────────┘
                       │
                 Keyframe Selection
                       │
                       ▼
              ┌─────────────────┐
              │     YOLOv8      │
              │    Detection    │
              └────────┬────────┘
                       │
              Detected Ingredients
                       │
                       ▼
          ┌─────────────────────────┐
          │ Sentence Transformers   │
          │   all-MiniLM-L6-v2      │
          └────────────┬────────────┘
                       │
                    Embedding
                       │
                       ▼
              ┌─────────────────┐
              │    Pinecone     │
              │ Vector Search   │
              └────────┬────────┘
                       │
              Relevant Recipes
                       │
                       ▼
              Ingredient Ranking
                       │
                       ▼
              ┌─────────────────┐
              │      Groq       │
              │  Recipe LLM     │
              └────────┬────────┘
                       │
                       ▼
                Generated Recipe
```

##  How It Works

### 1. Upload a Video

Upload a short `.mp4` or `.mov` video showing the contents of a refrigerator or kitchen counter.

### 2. Select Useful Frames

OpenCV processes the video and selects useful frames using:

* Motion-based frame selection
* Frame sampling fallback
* Duplicate/redundant frame reduction

### 3. Detect Ingredients

YOLOv8 analyzes the selected frames and identifies supported food and kitchen-object classes.

Detections are aggregated across multiple frames to improve confidence.

### 4. Create the Ingredient Query

The detected ingredients are converted into a text query and embedded using:

```text
all-MiniLM-L6-v2
```

### 5. Retrieve Recipes

The embedding is sent to Pinecone to retrieve semantically similar recipes from the `fridge-recipes` index.

Retrieved recipes are additionally ranked using ingredient overlap so that recipes requiring ingredients actually present in the video are prioritized.

### 6. Generate the Recipe

Groq generates a final recipe using:

* Detected ingredients
* Retrieved recipes
* Ingredient overlap
* A small set of common pantry staples

The generation is constrained to reduce the chance of suggesting ingredients that were not detected.

---

##  Tech Stack

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| **Python**                | Core application                      |
| **YOLOv8**                | Ingredient/object detection           |
| **OpenCV**                | Video processing & keyframe selection |
| **Sentence Transformers** | Text embeddings                       |
| **Pinecone**              | Vector database & semantic search     |
| **Groq**                  | LLM-powered recipe generation         |
| **Flask**                 | Web application & API                 |
| **NumPy / Pandas**        | Data processing                       |

---

##  Project Structure

```text
fridge_ai/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── recipe.csv
├── extracted_script.js
├── inspect_csv.py
├── seed.py
├── seed_kaggle.py
├── write_index.py
├── write_processing.py
├── test_connection.py
├── test_pinecone.py
├── test_video.py
├── test_yolo.py
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── processing.html
│   └── result.html
├── uploads/
│   └── .gitkeep
└── static/
    └── detections/
        └── .gitkeep                    # API credentials (not committed)
```

---

## Getting Started

### Requirements

* Python **3.10+**
* Pinecone account and API key
* Groq account and API key
* YOLOv8 model weights
* Sufficient disk space for PyTorch, Ultralytics and Sentence Transformers dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fridge-ai.git
cd fridge-ai
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

> **Never commit your `.env` file or expose API keys publicly.**

The application expects a Pinecone index named:

```text
fridge-recipes
```

with:

```text
Dimension: 384
Metric: cosine
```

---

##  YOLOv8 Model

The YOLO model weights are intentionally excluded from Git because of their size.

Download the required YOLOv8 model and place it in the project root:

```text
fridge-ai/
└── yolov8m.pt
```

Alternatively, update the model path in `app.py`.

The Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

is downloaded and cached automatically the first time it is used.

---

##  Seed the Recipe Database

### Option 1 — Sample Recipes

```bash
python seed.py
```

### Option 2 — CSV Recipe Dataset

```bash
python seed_kaggle.py
```

The Kaggle-style seed script samples up to **400 recipes**, generates embeddings, and stores them in Pinecone.

> Run only one seeding method unless you intentionally want to extend the existing index.

---

## Run the Application

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload a short refrigerator or kitchen video and let the AI pipeline process it.

---

## Testing

Individual components can be tested using:

```bash
python test_connection.py
python test_pinecone.py
python test_yolo.py
python test_video.py
```

Some tests require API credentials, model weights, or local test media.

---

##  Key Engineering Concepts

This project demonstrates several practical AI engineering concepts:

* Computer Vision
* Object Detection
* Video Processing
* Keyframe Selection
* Multi-frame Detection Aggregation
* Text Embeddings
* Vector Databases
* Semantic Search
* Hybrid Similarity + Metadata Ranking
* Retrieval-Augmented Generation
* LLM Prompt Constraining
* REST APIs
* AI Pipeline Integration

---

##  Limitations

* Detection accuracy depends on the model used.
