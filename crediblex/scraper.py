import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# This is first prototype of web scraper
# ------------------ Utilities ------------------
def to_iso_date(d):
    """Convert various date representations to ISO8601 string or return original if unknown."""
    if d is None:
        return None

    # Numeric epoch (ms or s)
    if isinstance(d, (int, float)) or (isinstance(d, str) and re.fullmatch(r"\d{10,16}", d.strip())):
        try:
            val = int(str(d).strip())
            # if looks like ms (13 digits) -> ms, if 10 digits -> s
            if val > 10**12:
                ts = val / 1000.0
            elif val > 10**9:
                ts = val
            else:
                ts = val
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            pass

    # ISO-like strings or other textual dates: try basic normalization
    if isinstance(d, str):
        # common ISO substring
        iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", d)
        if iso_match:
            return iso_match.group(0)
        # yyyy-mm-dd fallback
        ymd = re.search(r"\d{4}-\d{2}-\d{2}", d)
        if ymd:
            return ymd.group(0)
        # else return stripped string (best effort)
        return d.strip()

    # Unknown type -> string
    return str(d)


def join_authors(a):
    if not a:
        return None
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, dict):
        return a.get("name") or a.get("author") or None
    if isinstance(a, (list, tuple)):
        names = []
        for e in a:
            if isinstance(e, str):
                names.append(e.strip())
            elif isinstance(e, dict):
                name = e.get("name") or e.get("author")
                if name:
                    names.append(name.strip())
        return ", ".join([n for n in names if n])
    return None


def normalize_text_blocks(blocks):
    """Take a list of text blocks, flatten, dedupe (preserve order) and join with newlines."""
    if not blocks:
        return None
    out = []
    seen = set()
    for b in blocks:
        if not b:
            continue
        # split into lines and treat each meaningful line individually
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
            # skip very short noise lines
            if len(key) < 4:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
    if not out:
        return None
    # Remove top-of-text repetitions like title/author: if first lines include title or author duplicates,
    # we can drop them later when we have definitive title/author.
    return "\n".join(out)


# ------------------ JSON-LD extraction ------------------
def extract_from_json_ld(soup):
    scripts = soup.find_all("script", type="application/ld+json")
    candidates = []
    for s in scripts:
        try:
            raw = s.string
            if not raw:
                continue
            data = json.loads(raw)
            # normalize list case
            if isinstance(data, list):
                for item in data:
                    candidates.append(item)
            else:
                candidates.append(data)
        except Exception:
            continue

    # prefer NewsArticle or Article types
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        t = obj.get("@type") or obj.get("type")
        if t and ("NewsArticle" in t or "Article" in t):
            title = obj.get("headline") or obj.get("name") or None
            author = join_authors(obj.get("author") or obj.get("creator"))
            date = obj.get("datePublished") or obj.get("dateCreated") or obj.get("dateModified")
            body = obj.get("articleBody") or obj.get("description")
            return title, body, author, to_iso_date(date)

    # fallback: any ld+json with headline
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
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
        if not txt:
            continue
        # __NEXT_DATA__ typical
        if s.get("id") == "__NEXT_DATA__" or "__NEXT_DATA__" in (s.get("id") or ""):
            try:
                blobs.append(json.loads(txt))
            except Exception:
                # sometimes __NEXT_DATA__ is directly JSON; if contains '=' then try to extract
                try:
                    j = re.search(r"(\{.*\})", txt, flags=re.S)
                    if j:
                        blobs.append(json.loads(j.group(1)))
                except Exception:
                    pass
        # inline assignment patterns
        if "window.__INITIAL_STATE__" in txt or "window.__PRELOADED_STATE__" in txt or "window.__DATA__" in txt:
            try:
                # crude extraction of JSON after '='
                jmatch = re.search(r"=\s*({.*})\s*;", txt, flags=re.S)
                if jmatch:
                    blobs.append(json.loads(jmatch.group(1)))
            except Exception:
                pass
        # generic heuristic: if text looks like JSON with "props" and "pageProps"
        if '"props"' in txt and '"pageProps"' in txt:
            try:
                j = json.loads(txt)
                blobs.append(j)
            except Exception:
                try:
                    j = re.search(r"(\{.*\})", txt, flags=re.S)
                    if j:
                        blobs.append(json.loads(j.group(1)))
                except Exception:
                    pass

    # walk blobs for candidate fields
    def walk(obj, acc):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in ["headline", "title", "name"]:
                    if isinstance(v, str) and len(v) > 5:
                        acc.setdefault("titles", []).append(v)
                if key in ["byline", "author", "authors", "creator", "contributor"]:
                    acc.setdefault("authors", []).append(v)
                if "date" in key and isinstance(v, (str, int, float)):
                    acc.setdefault("dates", []).append(v)
                # body-like keys
                if key in ["articlebody", "body", "content", "text", "html", "article"]:
                    acc.setdefault("bodies", []).append(v)
                # common container names with nested blocks
                if key in ["blocks", "model", "contentblocks", "components", "children"]:
                    acc.setdefault("bodies", []).append(v)
                walk(v, acc)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, acc)
        else:
            return

    for blob in blobs:
        acc = {}
        walk(blob, acc)
        # build fields with best heuristics
        title = None
        if acc.get("titles"):
            # prefer the longest title
            title = max([t for t in acc["titles"] if isinstance(t, str)], key=len, default=None)
        author = None
        if acc.get("authors"):
            # flatten and join
            flattened = []
            for a in acc["authors"]:
                if isinstance(a, str):
                    flattened.append(a)
                elif isinstance(a, dict):
                    name = a.get("name") or a.get("firstName") or a.get("lastName")
                    if name:
                        flattened.append(name)
                elif isinstance(a, list):
                    for e in a:
                        if isinstance(e, str):
                            flattened.append(e)
                        elif isinstance(e, dict):
                            n = e.get("name")
                            if n:
                                flattened.append(n)
            if flattened:
                author = ", ".join(dict.fromkeys([x.strip() for x in flattened if x]))  # unique preserve order
        date = None
        if acc.get("dates"):
            # choose the first plausible date
            date = to_iso_date(acc["dates"][0])
        # collect text blocks
        bodies = []
        if acc.get("bodies"):
            for b in acc["bodies"]:
                if isinstance(b, str):
                    bodies.append(b)
                elif isinstance(b, list):
                    for e in b:
                        if isinstance(e, str):
                            bodies.append(e)
                        elif isinstance(e, dict):
                            # extract any text-like fields
                            for key in ["text", "body", "html", "content"]:
                                if key in e and isinstance(e[key], str):
                                    bodies.append(e[key])
                elif isinstance(b, dict):
                    # pull likely fields
                    for key in ["text", "body", "articleBody", "html", "content"]:
                        if key in b and isinstance(b[key], str):
                            bodies.append(b[key])
                    # sometimes blocks have 'model'->'blocks' nested
                    if "blocks" in b and isinstance(b["blocks"], list):
                        for blk in b["blocks"]:
                            if isinstance(blk, dict):
                                for k2 in ["text", "body", "html"]:
                                    if k2 in blk and isinstance(blk[k2], str):
                                        bodies.append(blk[k2])
        if bodies:
            text = normalize_text_blocks(bodies)
            return title, text, author, date

    return None, None, None, None


# ------------------ HTML fallback ------------------
def extract_from_html(soup):
    # Title: prefer og:title, then h1
    title = None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None

    # Author meta
    author = None
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        author = meta_author["content"].strip()
    if not author:
        # try common classes
        for cls in ["byline", "author", "by-line", "auth", "auth-nm", "by"]:
            tag = soup.find(class_=cls)
            if tag:
                author = tag.get_text(strip=True)
                break

    # Date meta
    date = None
    meta_date = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name": "publish-date"})
    if meta_date and meta_date.get("content"):
        date = to_iso_date(meta_date["content"])

    # Article text: try <article>, then common wrappers, then all <p>
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
        if tag:
            node = soup.find(tag)
        else:
            node = soup.find(class_=cls) if cls else None
        if node:
            ps = node.find_all("p")
            if ps:
                blocks = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]
                break

    if not blocks:
        # fallback: all paragraphs
        ps = soup.find_all("p")
        blocks = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]

    text = normalize_text_blocks(blocks)
    return title, text, author, date


# ------------------ Main scraper ------------------
def scrape_news(url):
    html = requests.get(url, headers=HEADERS, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    # 1) JSON-LD
    title, text, author, date = extract_from_json_ld(soup)
    # don't accept tiny junk
    if text and len(text.split()) > 40:
        text = _clean_title_author_in_text(text, title, author)
        return _finalize(title, text, author, date)

    # 2) JSON blobs (Next.js / initial state)
    title2, text2, author2, date2 = extract_from_json_blobs(soup)
    if text2 and len(text2.split()) > 40:
        text2 = _clean_title_author_in_text(text2, title2 or title, author2 or author)
        return _finalize(title2 or title, text2, author2 or author, date2 or date)

    # 3) HTML fallback
    title3, text3, author3, date3 = extract_from_html(soup)
    if text3 and len(text3.split()) > 20:
        text3 = _clean_title_author_in_text(text3, title3 or title2 or title, author3 or author2 or author)
        return _finalize(title3 or title2 or title, text3, author3 or author2 or author, date3 or date2 or date)

    # final fallback: return whatever we have
    combined_title = title or title2 or title3
    combined_text = text or text2 or text3
    combined_author = author or author2 or author3
    combined_date = date or date2 or date3
    if combined_text:
        combined_text = _clean_title_author_in_text(combined_text, combined_title, combined_author)
    return _finalize(combined_title, combined_text, combined_author, combined_date)


# ------------------ Cleaning helpers ------------------
def _clean_title_author_in_text(text, title, author):
    """Remove title/author duplicates at top of text if present."""
    if not text:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    # remove leading lines equal to title or author
    cleaned = list(lines)
    changed = False
    if title:
        tkey = re.sub(r"\s+", " ", title.strip().lower())
        while cleaned and re.sub(r"\s+", " ", cleaned[0].lower()) == tkey:
            cleaned.pop(0)
            changed = True
    if author:
        akey = re.sub(r"\s+", " ", author.strip().lower())
        while cleaned and re.sub(r"\s+", " ", cleaned[0].lower()) == akey:
            cleaned.pop(0)
            changed = True
    # remove repeated consecutive duplicates
    out = []
    prev = None
    for ln in cleaned:
        if prev and re.sub(r"\s+", " ", ln.lower()) == re.sub(r"\s+", " ", prev.lower()):
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out)


def _finalize(title, text, author, date):
    # normalize whitespace & trim extremely long repeated blocks
    if title:
        title = title.strip()
    if author:
        author = author.strip()
    if date:
        date = to_iso_date(date)
    if text:
        # remove sequences of the same sentence repeated many times (very crude)
        text = re.sub(r"(?m)^(.+)\n\1(\n\1)+", r"\1", text)
        # collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return title, text, author, date
if __name__ == "__main__":
    url = "https://www.thehindu.com/news/national/ed-raids-medical-colleges-for-bribes/article70329264.ece"
    title, text, author, date = scrape_news(url)

    print("TITLE:", title)
    print("AUTHOR:", author)
    print("DATE:", date)
    print("TEXT:", text[:])
