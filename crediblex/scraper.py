import json, re
from curl_cffi import requests
from typing import Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel, field_validator, ValidationError
from datetime import datetime, timezone

class NewsArticle(BaseModel):
    title:          Optional[str] = None
    text:           str
    url:            Optional[str] = None
    author:         Optional[str] = None
    published_date: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_must_have_content(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 50:
            raise ValueError("Extracted text too short (< 50 chars)")
        return v

# ------------------ Utilities ------------------
def to_iso_date(d):
    """Convert various date representations to ISO8601 string or return original if unknown."""
    if d is None:
        return None

    # Numeric epoch (ms or s)
    if isinstance(d, (int, float)) or (isinstance(d, str) and re.fullmatch(r"\d{10,16}", d.strip())):
        try:
            val = int(str(d).strip())
            if val > 10**12: ts = val / 1000.0
            elif val > 10**9: ts = val
            else: ts = val
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            pass

    # ISO-like strings or other textual dates
    if isinstance(d, str):
        iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", d)
        if iso_match: return iso_match.group(0)
        ymd = re.search(r"\d{4}-\d{2}-\d{2}", d)
        if ymd: return ymd.group(0)
        return d.strip()

    return str(d)

def join_authors(a):
    if not a: return None
    if isinstance(a, str): return a.strip()
    if isinstance(a, dict): return a.get("name") or a.get("author") or None
    if isinstance(a, (list, tuple)):
        names = []
        for e in a:
            if isinstance(e, str):
                names.append(e.strip())
            elif isinstance(e, dict):
                name = e.get("name") or e.get("author")
                if name: names.append(name.strip())
        return ", ".join([n for n in names if n])
    return None

def normalize_text_blocks(blocks):
    """Take a list of text blocks, flatten, dedupe (preserve order) and join with newlines."""
    if not blocks: return None
    out = []
    seen = set()
    for b in blocks:
        if not b: continue
        if isinstance(b, str):
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        elif isinstance(b, (list, tuple)):
            lines = []
            for item in b:
                if isinstance(item, str):
                    lines.extend([ln.strip() for ln in item.splitlines() if ln.strip()])
        else:
            lines = [str(b).strip()]

        for ln in lines:
            key = re.sub(r"\s+", " ", ln.lower()).strip()
            if len(key) < 4 or key in seen: continue
            seen.add(key)
            out.append(ln)
    if not out: return None
    return "\n".join(out)

# ------------------ JSON-LD extraction ------------------
def extract_from_json_ld(soup):
    scripts = soup.find_all("script", type="application/ld+json")
    candidates = []
    for s in scripts:
        try:
            raw = s.string
            if not raw: continue
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data: candidates.append(item)
            else:
                candidates.append(data)
        except Exception:
            continue

    for obj in candidates:
        if not isinstance(obj, dict): continue
        t = obj.get("@type") or obj.get("type")
        if t and ("NewsArticle" in t or "Article" in t):
            title = obj.get("headline") or obj.get("name") or None
            author = join_authors(obj.get("author") or obj.get("creator"))
            date = obj.get("datePublished") or obj.get("dateCreated") or obj.get("dateModified")
            body = obj.get("articleBody") or obj.get("description")
            return title, body, author, to_iso_date(date)

    for obj in candidates:
        if not isinstance(obj, dict): continue
        if "headline" in obj:
            title = obj.get("headline")
            author = join_authors(obj.get("author"))
            date = obj.get("datePublished") or obj.get("dateModified")
            body = obj.get("articleBody") or obj.get("description")
            return title, body, author, to_iso_date(date)

    return None, None, None, None

# ------------------ Next.js / other JSON-blob extraction ------------------
def extract_from_json_blobs(soup):
    scripts = soup.find_all("script")
    blobs = []
    for s in scripts:
        txt = s.string
        if not txt: continue
        if s.get("id") == "__NEXT_DATA__" or "__NEXT_DATA__" in (s.get("id") or ""):
            try: blobs.append(json.loads(txt))
            except Exception:
                try:
                    j = re.search(r"(\{.*\})", txt, flags=re.S)
                    if j: blobs.append(json.loads(j.group(1)))
                except Exception: pass
        if "window.__INITIAL_STATE__" in txt or "window.__PRELOADED_STATE__" in txt or "window.__DATA__" in txt:
            try:
                jmatch = re.search(r"=\s*({.*})\s*;", txt, flags=re.S)
                if jmatch: blobs.append(json.loads(jmatch.group(1)))
            except Exception: pass
        if '"props"' in txt and '"pageProps"' in txt:
            try: blobs.append(json.loads(txt))
            except Exception:
                try:
                    j = re.search(r"(\{.*\})", txt, flags=re.S)
                    if j: blobs.append(json.loads(j.group(1)))
                except Exception: pass

    def walk(obj, acc):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in ["headline", "title", "name"]:
                    if isinstance(v, str) and len(v) > 5: acc.setdefault("titles", []).append(v)
                if key in ["byline", "author", "authors", "creator", "contributor"]:
                    acc.setdefault("authors", []).append(v)
                if "date" in key and isinstance(v, (str, int, float)):
                    acc.setdefault("dates", []).append(v)
                if key in ["articlebody", "body", "content", "text", "html", "article"]:
                    acc.setdefault("bodies", []).append(v)
                if key in ["blocks", "model", "contentblocks", "components", "children"]:
                    acc.setdefault("bodies", []).append(v)
                walk(v, acc)
        elif isinstance(obj, list):
            for item in obj: walk(item, acc)

    for blob in blobs:
        acc = {}
        walk(blob, acc)
        title = max([t for t in acc["titles"] if isinstance(t, str)], key=len, default=None) if acc.get("titles") else None
        author = None
        if acc.get("authors"):
            flattened = []
            for a in acc["authors"]:
                if isinstance(a, str): flattened.append(a)
                elif isinstance(a, dict):
                    name = a.get("name") or a.get("firstName") or a.get("lastName")
                    if name: flattened.append(name)
                elif isinstance(a, list):
                    for e in a:
                        if isinstance(e, str): flattened.append(e)
                        elif isinstance(e, dict):
                            n = e.get("name")
                            if n: flattened.append(n)
            if flattened: author = ", ".join(dict.fromkeys([x.strip() for x in flattened if x]))
        date = to_iso_date(acc["dates"][0]) if acc.get("dates") else None
        
        bodies = []
        if acc.get("bodies"):
            for b in acc["bodies"]:
                if isinstance(b, str): bodies.append(b)
                elif isinstance(b, list):
                    for e in b:
                        if isinstance(e, str): bodies.append(e)
                        elif isinstance(e, dict):
                            for key in ["text", "body", "html", "content"]:
                                if key in e and isinstance(e[key], str): bodies.append(e[key])
                elif isinstance(b, dict):
                    for key in ["text", "body", "articleBody", "html", "content"]:
                        if key in b and isinstance(b[key], str): bodies.append(b[key])
                    if "blocks" in b and isinstance(b["blocks"], list):
                        for blk in b["blocks"]:
                            if isinstance(blk, dict):
                                for k2 in ["text", "body", "html"]:
                                    if k2 in blk and isinstance(blk[k2], str): bodies.append(blk[k2])
        if bodies:
            text = normalize_text_blocks(bodies)
            return title, text, author, date

    return None, None, None, None

# ------------------ HTML fallback ------------------
def extract_from_html(soup):
    title = None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"): title = og["content"].strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None

    author = None
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"): author = meta_author["content"].strip()
    if not author:
        for cls in ["byline", "author", "by-line", "auth", "auth-nm", "by"]:
            tag = soup.find(class_=cls)
            if tag:
                author = tag.get_text(strip=True)
                break

    date = None
    meta_date = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name": "publish-date"})
    if meta_date and meta_date.get("content"): date = to_iso_date(meta_date["content"])

    text = None
    candidates = [
        ("article", None),
        (None, "article-body"),
        (None, "story-content"),
        (None, "content"),
        (None, "detail__body-txt"),
        (None, "description"),
        (None, "live-blog-wrapper"),
    ]
    blocks = []
    for tag, cls in candidates:
        node = soup.find(tag) if tag else (soup.find(class_=cls) if cls else None)
        if node:
            ps = node.find_all("p")
            if ps:
                blocks = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]
                break

    if not blocks:
        ps = soup.find_all("p")
        blocks = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]

    text = normalize_text_blocks(blocks)
    return title, text, author, date

# ------------------ Cleaning helpers ------------------
def _clean_title_author_in_text(text, title, author):
    if not text: return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return text
    cleaned = list(lines)
    if title:
        tkey = re.sub(r"\s+", " ", title.strip().lower())
        while cleaned and re.sub(r"\s+", " ", cleaned[0].lower()) == tkey: cleaned.pop(0)
    if author:
        akey = re.sub(r"\s+", " ", author.strip().lower())
        while cleaned and re.sub(r"\s+", " ", cleaned[0].lower()) == akey: cleaned.pop(0)
    out = []
    prev = None
    for ln in cleaned:
        if prev and re.sub(r"\s+", " ", ln.lower()) == re.sub(r"\s+", " ", prev.lower()): continue
        out.append(ln)
        prev = ln
    return "\n".join(out)

def _finalize(title, text, author, date):
    if title: title = title.strip()
    if author: author = author.strip()
    if date: date = to_iso_date(date)
    if text:
        text = re.sub(r"(?m)^(.+)\n\1(\n\1)+", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, author, date

# ------------------ Boilerplate / noise filtering ------------------
BOILERPLATE_PATTERNS = [
    re.compile(r"^(updated|published)\s*[-:]?\s*\w+\s+\d", re.I),
    re.compile(r"premium\s*article", re.I),
    re.compile(r"exclusively\s*to\s*subscribers", re.I),
    re.compile(r"this\s*offer\s*is\s*only", re.I),
    re.compile(r"subscribe\s*(now|to\s*continue)", re.I),
    re.compile(r"^(share|tweet|email|print|copy link)\s*$", re.I),
    re.compile(r"^advertisement\s*$", re.I),
    re.compile(r"^(also read|read more|related)\s*:", re.I),
    re.compile(r"^\d+\s*(min|minute)\s*read$", re.I),
    re.compile(r"active subscription", re.I),
    re.compile(r"unlock\s*(these|with)\s*subscription", re.I),
    re.compile(r"login to read|sign in to read", re.I),
    re.compile(r"independent,?\s*credible\s*journalism", re.I),
    re.compile(r"^source:\s*", re.I),
]

def _filter_boilerplate(text):
    """Remove common boilerplate lines from article text."""
    if not text: return text
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if any(pat.search(stripped) for pat in BOILERPLATE_PATTERNS): continue
        cleaned.append(stripped)
    return "\n".join(cleaned) if cleaned else None

# ------------------ Paywall detection ------------------
PAYWALL_SIGNALS = [
    "active subscription", "subscribe to continue", "subscribe now",
    "you don't have any active", "unlock with subscription", "login to read",
    "sign in to read", "please subscribe", "premium content", "to continue reading",
    # Indian news sites (The Hindu, Indian Express, etc.)
    "unlock these with subscription", "subscription benefits",
    "subscribed with another email", "need help with your subscription",
    "e-paper", "additional subscription benefits",
    # Generic paywalls
    "create a free account", "register to continue", "log in to continue",
    "become a member", "members-only", "subscriber-only",
    "start your free trial", "already a subscriber",
]

PAYWALL_PATTERNS = [
    re.compile(r"unlock\s+\w+\s+with\s+subscription", re.I),
    re.compile(r"subscription\s+benefits", re.I),
    re.compile(r"subscribed\s+with\s+another\s+email", re.I),
    re.compile(r"log\s*out\s+and\s+log\s*in", re.I),
]

def _is_paywalled(text):
    """Check if extracted text looks like a paywall page rather than real content."""
    if not text: return True
    text_lower = text.lower()
    signal_count = sum(1 for s in PAYWALL_SIGNALS if s in text_lower)
    pattern_count = sum(1 for p in PAYWALL_PATTERNS if p.search(text))
    total = signal_count + pattern_count
    # If 2+ paywall signals found, it's almost certainly a paywall page
    if total >= 2: return True
    # If only 1 signal but the real content is very short, still treat as paywalled
    if total >= 1 and len(text.split()) < 100: return True

    return False

# ------------------ AMP fallback ------------------
def _try_amp_fallback(url):
    """
    Many paywalled Indian news sites (The Hindu, Indian Express, etc.) serve
    the full article body on their AMP pages. This function tries to fetch
    the AMP version and extract the article content from it.
    """
    # Build candidate AMP URLs
    amp_urls = []
    if ".ece" in url:
        amp_urls.append(re.sub(r"\.ece/?$", ".ece/amp/", url))
    if "/amp" not in url:
        amp_urls.append(url.rstrip("/") + "/amp/")
        amp_urls.append(url.rstrip("/") + "?amp=1")

    # Use a mobile user-agent for AMP pages
    mobile_ua = "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36"

    for amp_url in amp_urls:
        try:
            import requests as fallback_requests
            resp = fallback_requests.get(amp_url, headers={"User-Agent": mobile_ua}, timeout=12)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Extract title from AMP page
            title = None
            og = soup.find("meta", property="og:title")
            if og and og.get("content"): title = og["content"].strip()
            if not title:
                h1 = soup.find("h1")
                if h1: title = h1.get_text(strip=True)

            # Extract author
            author = None
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author and meta_author.get("content"):
                author = meta_author["content"].strip()
            if not author:
                for cls in ["author", "byline", "by-line"]:
                    tag = soup.find(class_=lambda c: c and cls in ' '.join(c).lower() if c else False)
                    if tag:
                        author = tag.get_text(strip=True)
                        break

            # Extract date
            date = None
            meta_date = soup.find("meta", property="article:published_time")
            if meta_date and meta_date.get("content"):
                date = to_iso_date(meta_date["content"])
            if not date:
                time_tag = soup.find("time")
                if time_tag and time_tag.get("datetime"):
                    date = to_iso_date(time_tag["datetime"])

            # Extract article body from AMP
            # Try <article>, then common wrappers, then all <p>
            text_parts = []
            article_node = soup.find("article") or soup.find(class_=lambda c: c and "article" in ' '.join(c).lower() if c else False)
            if article_node:
                for p in article_node.find_all("p"):
                    t = p.get_text(" ", strip=True)
                    if len(t) > 25: text_parts.append(t)
            if not text_parts:
                for p in soup.find_all("p"):
                    t = p.get_text(" ", strip=True)
                    if len(t) > 25: text_parts.append(t)

            if text_parts:
                raw_text = "\n".join(text_parts)
                cleaned = _filter_boilerplate(raw_text)
                if cleaned and len(cleaned.split()) > 40:
                    title_final, text_final, author_final, date_final = _finalize(
                        title, cleaned, author, date
                    )
                    if text_final:
                        return title_final, text_final, author_final, date_final

        except Exception:
            continue

    return None, None, None, None

# ------------------ Main scraper ------------------
def extract_article(url: str) -> Optional[dict]:
    try:
        response = requests.get(url, impersonate="chrome110", timeout=15, allow_redirects=True)
        if response.status_code != 200:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # ── Strategy 1: JSON-LD ──
    title1, text1, author1, date1 = extract_from_json_ld(soup)
    if text1 and len(text1.split()) > 40:
        text1 = _clean_title_author_in_text(text1, title1, author1)
        title, text, author, date = _finalize(title1, text1, author1, date1)
        if text and not _is_paywalled(text):
            text = _filter_boilerplate(text)
            if text:
                return _build_result(title, text, author, date, url)

    # ── Strategy 2: JSON blobs (Next.js / __INITIAL_STATE__) ──
    title2, text2, author2, date2 = extract_from_json_blobs(soup)
    if text2 and len(text2.split()) > 40:
        text2 = _clean_title_author_in_text(text2, title2 or title1, author2 or author1)
        title, text, author, date = _finalize(title2 or title1, text2, author2 or author1, date2 or date1)
        if text and not _is_paywalled(text):
            text = _filter_boilerplate(text)
            if text:
                return _build_result(title, text, author, date, url)

    # ── Strategy 3: HTML fallback ──
    title3, text3, author3, date3 = extract_from_html(soup)
    combined_title = title3 or title2 or title1
    combined_author = author3 or author2 or author1
    combined_date = date3 or date2 or date1
    combined_text = text3 or text2 or text1
    if combined_text:
        combined_text = _clean_title_author_in_text(combined_text, combined_title, combined_author)
    title, text, author, date = _finalize(combined_title, combined_text, combined_author, combined_date)

    if text and not _is_paywalled(text):
        text = _filter_boilerplate(text)
        if text:
            return _build_result(title, text, author, date, url)

    # ── Strategy 4: AMP fallback (for paywalled sites) ──
    amp_title, amp_text, amp_author, amp_date = _try_amp_fallback(url)
    if amp_text:
        # Prefer metadata from the main page if AMP didn't find them
        final_title = amp_title or title or combined_title
        final_author = amp_author or author or combined_author
        final_date = amp_date or date or combined_date
        return _build_result(final_title, amp_text, final_author, final_date, url)

    # ── Strategy 5: og:description as last resort ──
    og_desc = soup.find("meta", property="og:description")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    desc = None
    if og_desc and og_desc.get("content"): desc = og_desc["content"].strip()
    elif meta_desc and meta_desc.get("content"): desc = meta_desc["content"].strip()
    if desc and len(desc) >= 50:
        final_title = title or combined_title
        return _build_result(final_title, desc, author or combined_author, date or combined_date, url)

    return None

def _build_result(title, text, author, date, url):
    """Validate and build the final result dict."""
    try:
        article_obj = NewsArticle(
            title=title, text=text, url=url,
            author=author, published_date=date
        )
        return article_obj.model_dump()
    except Exception:
        return None

if __name__ == "__main__":
    test_url = "https://www.bbc.com/news/world"
    print(f"Testing scraper on: {test_url}")
    result = extract_article(test_url)
    if result:
        print(f"  Title:   {result['title'][:60] if result['title'] else 'None'}")
        print(f"  Text:    {len(result['text'])} chars")
        print(f"  Author:  {result['author']}")
        print("Part E scraper.py validation passed")
    else:
        print("  Extraction returned None (network/structure issue)")