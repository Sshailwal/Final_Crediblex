\# CredibleX 🛡️



> \*\*AI-powered news credibility analyzer\*\* — paste a URL or WhatsApp message and get a full trust report in seconds.



!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python\&logoColor=white)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi\&logoColor=white)

!\[React](https://img.shields.io/badge/React-19-61DAFB?logo=react\&logoColor=black)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch\&logoColor=white)

!\[License](https://img.shields.io/badge/License-MIT-green)



\---



\## 📌 What is CredibleX?



CredibleX is a group project that analyzes news articles and viral messages for credibility. It uses a fine-tuned \*\*DeBERTa-v3-base\*\* transformer model to score content across four key dimensions:



| Dimension | What it measures |

|-----------|-----------------|

| 🧾 \*\*Factuality\*\* | How factually accurate is the content? |

| ⚖️ \*\*Political Bias\*\* | Left / Center / Right leaning? |

| 🎯 \*\*Intent\*\* | News, Opinion, or Satire? |

| 😤 \*\*Emotion\*\* | What emotional tone does it carry? |



A final \*\*Trust Score (0–100)\*\* and verdict is generated based on these dimensions.



\---



\## ✨ Features



\- 🔗 \*\*URL Analysis\*\* — Paste any public news article URL and get a full credibility report

\- 💬 \*\*WhatsApp / Text Analysis\*\* — Paste raw text (forwards, messages) for fact-checking

\- 📊 \*\*Trust Score Gauge\*\* — Visual score from 0–100 with color-coded verdict

\- 📋 \*\*Key Findings\*\* — Bullet summary of the most important signals

\- ⚡ \*\*Fast Inference\*\* — Model is loaded once and cached per process



\---



\## 🏗️ Project Structure



```

CredibleX/

├── crediblex/                  # Main application

│   ├── api.py                  # FastAPI server (endpoints: /analyze, /analyze-text, /health, /logs)

│   ├── inference.py            # Model inference logic

│   ├── model.py                # PyTorch model definition (NewsTrustModel)

│   ├── train.py                # Model training script

│   ├── evaluate.py             # Model evaluation script

│   ├── scraper.py              # Article scraper (BeautifulSoup4)

│   ├── data\_ingest.py          # Dataset ingestion pipeline

│   ├── config.py               # Hyperparameters and paths

│   ├── schema.py               # Pydantic schemas

│   ├── requirements.txt        # Python dependencies

│   └── frontend/               # React + Vite frontend

│       ├── src/

│       │   ├── App.jsx         # Main app component

│       │   ├── components/

│       │   │   ├── UrlInput.jsx     # Dual-mode input (URL / Text)

│       │   │   ├── TrustGauge.jsx   # Animated score gauge

│       │   │   ├── BiasSlider.jsx   # Political bias slider

│       │   │   └── Badges.jsx       # Factuality, Intent, Emotion chips

│       │   └── index.css       # Global styles

│       └── package.json

└── ajay\_sentiment\_done/        # Sentiment model experiments (member work)

```



\---



\## 🚀 Getting Started



\### Prerequisites



\- Python 3.10+

\- Node.js 18+

\- CUDA-capable GPU (recommended) or CPU



\---



\### 1. Clone the Repository



```bash

git clone https://github.com/Harshitpandey21/CredibleX.git

cd CredibleX/crediblex

```



\---



\### 2. Set Up the Backend



```bash

\# Create and activate a virtual environment

python -m venv venv



\# On Windows:

venv\\Scripts\\activate



\# On Mac/Linux:

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt

```



\---



\### 3. Train the Model (First-time setup)



> ⚠️ A trained model file (`model\_v1.pth`) is required before the API can analyze articles. Skip this step if you already have the weights file.



```bash

python train.py

```



Training takes approximately \*\*6–8 hours\*\* on an RTX 4050 GPU (5 epochs, effective batch size 24).



\---



\### 4. Start the Backend API



```bash

python api.py

\# or

uvicorn api:app --reload --port 8000

```



The API will be available at: `http://127.0.0.1:8000`



\*\*API Endpoints:\*\*



| Method | Endpoint | Description |

|--------|----------|-------------|

| `GET` | `/health` | Check API status and model info |

| `POST` | `/analyze` | Analyze a news article URL |

| `POST` | `/analyze-text` | Fact-check raw text / WhatsApp message |

| `GET` | `/logs` | View recent analysis request logs |



\---



\### 5. Start the Frontend



Open a new terminal:



```bash

cd crediblex/frontend

npm install

npm run dev

```



The app will be available at: `http://localhost:5173`



> Make sure the backend is running on port `8000` before using the frontend.



\---



\## 🧪 Testing



```bash

\# Quick smoke test (backend)

python smoke\_test.py



\# Check GPU availability

python test\_gpu.py

```



\---



\## 🤖 Model Details



| Property | Value |

|----------|-------|

| Base Model | `microsoft/deberta-v3-base` |

| Max Token Length | 512 |

| Training Epochs | 5 |

| Learning Rate | 2e-5 |

| Batch Size | 4 (effective: 24 with gradient accumulation) |

| Device | CUDA (auto-detected) / CPU fallback |



The model performs \*\*multi-task classification\*\* simultaneously predicting:

\- Factuality score (regression, 0–1)

\- Political bias (3-class: Left / Center / Right)

\- Intent (3-class: News / Opinion / Satire)

\- Emotion (28-class GoEmotions labels)



\---



\## 🛠️ Tech Stack



\*\*Backend\*\*

\- \[FastAPI](https://fastapi.tiangolo.com/) — REST API framework

\- \[PyTorch](https://pytorch.org/) — Deep learning

\- \[HuggingFace Transformers](https://huggingface.co/docs/transformers) — DeBERTa model

\- \[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — Article scraping

\- \[Uvicorn](https://www.uvicorn.org/) — ASGI server



\*\*Frontend\*\*

\- \[React 19](https://react.dev/) — UI framework

\- \[Vite](https://vitejs.dev/) — Build tool

\- Vanilla CSS — Styling (no external UI library)



\---



\## 👥 Team



This is a group project. Contributions are tracked via GitHub commits and pull requests.



\---



\## 📄 License



This project is licensed under the MIT License.

