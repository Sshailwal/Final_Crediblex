You are an AI assistant tasked with understanding, maintaining, and extending a project called **CredibleX**. 

### 📌 Project Overview
CredibleX is an AI-powered news credibility analyzer. It allows users to paste a URL or a raw text message (like a WhatsApp forward) and generates a comprehensive credibility report. It evaluates content across four dimensions: Factuality, Political Bias, Intent, and Emotion, ultimately providing an aggregated "Trust Score" from 0 to 100.

### 🤖 Model Architecture
- **Base Model:** `microsoft/deberta-v3-base`
- **Architecture:** It uses a multi-task learning approach with a custom classification head built on top of the base DeBERTa transformer.
- **Tasks:**
  - **Factuality:** Regression (Score from 0.0 to 1.0)
  - **Political Bias:** 3-class Classification (Left, Center, Right)
  - **Intent:** 3-class Classification (News, Opinion, Satire)
  - **Emotion:** Multi-label Classification (28 classes based on the GoEmotions taxonomy)

### 📊 Datasets Used
The training data is an aggregation of multiple specialized datasets to capture different signals. These are fetched and balanced via the `data_ingest.py` script:
1. **Factuality:** `UKPLab/liar` 
2. **Political Bias:** `mediabiasgroup/mbib-base` (Split: political_bias)
3. **Hyperpartisan Signal:** `zapsdcn/hyperpartisan_news`
4. **Satire/Intent:** `GonzaloA/fake_news`
5. **Emotion:** `google-research-datasets/go_emotions` (Simplified)

### 🛠️ Tech Stack
- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **AI/ML:** PyTorch, HuggingFace Transformers
- **Scraping:** BeautifulSoup4 (for extracting text from URLs)
- **Frontend:** React 19, Vite, Vanilla CSS

### ⚙️ Workflow & System Architecture
1. **Data Pipeline (`data_ingest.py`):** Fetches the datasets mentioned above, normalizes their labels into a unified format (defined in `schema.ArticleRecord`), upsamples minority bias classes to prevent class imbalance, and exports a unified `training_data.csv`.
2. **Model Training (`train.py`, `model.py` & `config.py`):** Loads the dataset, tokenizes the text using the DeBERTa tokenizer (max length 192), and trains the multi-task model using mixed precision and gradient accumulation. It saves the best weights locally (e.g., `best_bias_acc.pth`).
3. **Inference Server (`api.py` & `inference.py`):** 
   - A FastAPI server loads the trained weights into memory upon startup.
   - The `/analyze` endpoint takes a URL, scrapes its content using `scraper.py`, and passes the text to the model.
   - The `/analyze-text` endpoint accepts raw text directly (useful for WhatsApp forwards).
   - Both endpoints return a JSON response containing the factuality score, bias, intent, emotion labels, an aggregated `trust_score`, and dynamic verdicts.
4. **User Interface (`frontend/`):** A React application that communicates with the FastAPI endpoints to display animated trust gauges, bias sliders, and credibility badges based on the model's output.

### 🎯 Instructions for the AI
When asked to modify or debug this project, ensure you:
- **Respect the multi-task architecture:** Changing one classification head's loss weight or structure impacts the others. Always reference `config.py` for training hyperparameters.
- **Optimize for VRAM:** Keep VRAM usage in check (e.g., maintaining gradient accumulation, reasonable batch sizes, and gradient checkpointing as defined in `config.py`).
- **Use existing schemas:** Follow the established Pydantic schemas in `schema.py` for API requests and responses.
- **Maintain Frontend Simplicity:** Maintain the Vite+React frontend architecture using Vanilla CSS without introducing heavy external UI libraries unless explicitly requested by the user.
