from text_preprocessing import TextPreprocessor

tp = TextPreprocessor()

with open("raw_scraped_article.txt", "r") as f:
    raw = f.read()

clean = tp.preprocess(raw)
print(clean)
