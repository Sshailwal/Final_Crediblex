import re
import unicodedata

class TextPreprocessor:
    def __init__(self):
        # patterns to cut out completely
        self.remove_blocks = [
            r"Copyright.+?All rights reserved\.?",
            r"BACK TO TOP",
            r"Terms & conditions.*",
            r"Comments? have to be in English.*",
            r"We have migrated to a new commenting platform.*",
            r"If you do not have an account please register.*",
            r"Users can access their older comments by logging.*",
            r"This article is part of our Premium service.*",
            r"Subscribe now to get unlimited access.*",
            r"(corruption|bribery|fraud|economic offence|tax evasion|Delhi|New Delhi)\s*/\s*",
        ]

        # short non-useful lines (SEO/ads/navigation)
        self.useless_exact = {
            "advertisement", "ad", "home", "news", "india", "world",
            "premium", "politics", "business", "sports", "login", "subscribe"
        }

    def normalize(self, text: str) -> str:
        """Unicode normalization + remove weird chars"""
        text = unicodedata.normalize("NFKC", text)
        return text

    def strip_noise_blocks(self, text: str) -> str:
        for pattern in self.remove_blocks:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", " ", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def remove_useless_single_lines(self, text: str) -> str:
        cleaned_lines = []
        for line in text.splitlines():
            l = line.strip().lower()
            if len(l) <= 2:
                continue
            if l in self.useless_exact:
                continue
            if l.replace(" ", "") in self.useless_exact:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def clean_spaces(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess(self, text: str) -> str:
        """Main pipeline"""
        text = self.normalize(text)
        text = self.strip_noise_blocks(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_useless_single_lines(text)

        # keep only basic punctuation + letters + numbers
        text = re.sub(r"[^A-Za-z0-9.,!?;:\-\n ]", " ", text)

        text = self.clean_spaces(text)
        return text
