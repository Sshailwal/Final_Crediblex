# Rediblex Project Roadmap

## V1 — Released ✅
List of everything that was successfully built and stabilized for V1:
- Multi-task DeBERTa architecture (4 heads)
- Round-robin task sampler (gradient conflict resolved)
- Null-safe loss computation (ignore_index=-1, sentinel vectors)
- Per-sentence BABE bias labels (outlet noise removed)
- Back-translation augmentation pipeline
- HuberLoss factuality scoring
- BCE pos_weight emotion fix
- Full training pipeline with per-task go/no-go gates

## V2 — Bias + Emotion Improvements
Priority order for the next iteration:

1. **Bias accuracy >80%**
    - Scrape sentence-level annotations from AllSides.com directly
    - Use Claude or GPT-4 API to generate 500 synthetic sentences per class with explicit political framing
    - Fine-tune on SemEval 2019 Task 4 (hyperpartisan news)
        - mapping: hyperpartisan=Right, non-hyperpartisan=Center/Left
    - Target: 2,000+ rows per class, >80% held-out accuracy

2. **Emotion — activate Fear, Surprise, Disgust**
    - Source 300+ samples per missing class from GoEmotions full split
    - Fine-tune emotion head only for 3 epochs with new data
    - Target: 7/7 classes active at threshold=0.3

3. **Factuality calibration**
    - Add Platt scaling or isotonic regression post-processing
    - Target: MAE < 0.15, reliable range 0.0–1.0

## V3 — Multilingual + Domain Expansion
- Replace DeBERTa-base with DeBERTa-v3-base (better cross-lingual capabilities)
- Add Arabic and Spanish bias training data
- Test on social media text (Twitter/Reddit domain shift)
- Add a credibility head (source-level reputation signal)
- Distill to a smaller model for API serving (<200MB checkpoint)
