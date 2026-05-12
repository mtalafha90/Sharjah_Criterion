import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://astronomycenter.net/"
INDEX_URL = urljoin(BASE, "res.html")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CrescentResearchBot/1.0)"
}

MONTH_LINK_RE = re.compile(r"/icop/.*\.html$", re.I)

DATE_RE = re.compile(
    r"(الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت|الأحد)?\s*"
    r"(\d{1,2})\s+([^\d\n]+?)\s+(\d{4})"
)

OBS_BLOCK_START_RE = re.compile(r"^\s*\d+\s*\.\s*وقت الرصد:", re.M)

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return BeautifulSoup(r.text, "html.parser")

def extract_month_links():
    soup = get_soup(INDEX_URL)

    hijri_months = {
        "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
        "جمادى الأولى", "جمادى الآخرة", "رجب", "شعبان",
        "رمضان", "شوال", "ذو القعدة", "ذو الحجة"
    }

    links = []
    for a in soup.find_all("a", href=True):
        text = normalize_space(a.get_text(" ", strip=True))
        href = a["href"].strip()
        full = urljoin(INDEX_URL, href)

        if text in hijri_months:
            links.append({"label": text, "url": full})

    # deduplicate while preserving order
    seen = set()
    out = []
    for x in links:
        if x["url"] not in seen:
            seen.add(x["url"])
            out.append(x)

    return out

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def parse_bool_seen(text: str, phrase_yes: str, phrase_no: str):
    if phrase_yes in text:
        return True
    if phrase_no in text:
        return False
    return None

def extract_field(pattern: str, text: str):
    m = re.search(pattern, text)
    return normalize_space(m.group(1)) if m else None

def split_sections(full_text: str):
    # crude split around headings commonly used on the pages
    first_idx = full_text.find("نتائج تحري هلال أول الشهر")
    last_idx = full_text.find("نتائج تحري هلال آخر الشهر")
    parts = []
    if first_idx != -1:
        end = last_idx if last_idx != -1 else len(full_text)
        parts.append(("first_crescent", full_text[first_idx:end]))
    if last_idx != -1:
        parts.append(("last_crescent", full_text[last_idx:]))
    if not parts:
        parts.append(("unknown", full_text))
    return parts

def parse_country_blocks(section_text: str):
    """
    Split section by country headings like:
    ## الأردن
    """
    lines = section_text.splitlines()
    blocks = []
    current_country = None
    current = []
    for line in lines:
        s = line.strip()
        if s.startswith("## ") and "نتائج" not in s and len(s) < 80:
            if current_country and current:
                blocks.append((current_country, "\n".join(current)))
            current_country = s.replace("##", "").strip()
            current = []
        else:
            if current_country:
                current.append(line)
    if current_country and current:
        blocks.append((current_country, "\n".join(current)))
    return blocks

def split_observation_entries(country_text: str):
    starts = list(re.finditer(r"(?:^|\n)\s*\d+\s*\.\s*وقت الرصد:", country_text))
    if not starts:
        return []

    entries = []
    for i, m in enumerate(starts):
        start = m.start()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(country_text)
        entries.append(country_text[start:end].strip())
    return entries

def parse_entry(page_url, hijri_year, hijri_month_name, section_type, gregorian_date, country, entry_text):
    text = normalize_space(entry_text)

    obs_time_text = extract_field(r"وقت الرصد:\s*([^\.]+)", text)
    if obs_time_text:
        if "قبل غروب الشمس" in obs_time_text:
            obs_relative_to_sunset = "before_sunset"
        elif "بعد غروب الشمس" in obs_time_text:
            obs_relative_to_sunset = "after_sunset"
        else:
            obs_relative_to_sunset = "unknown"
    else:
        obs_relative_to_sunset = "unknown"

    city = extract_field(r"من مدينة\s+(.+?)\s+في محافظة", text)
    province = extract_field(r"في محافظة\s+(.+?)\s+ان السماء", text)

    sky_cloud_text = extract_field(r"ان السماء\s*\(نسبة الغيم\)\s*كانت\s+(.+?)\s+والحالة الجوية", text)
    weather_text = extract_field(r"والحالة الجوية كانت\s+(.+?)\s+(ولم|وتم|وقد)", text)

    seen_any = None
    if "تمت رؤية الهلال" in text or "تمت رؤية الهلال:" in text:
        seen_any = True
    elif "لم تتم رؤية الهلال" in text:
        seen_any = False

    seen_naked_eye = parse_bool_seen(text, "تمت رؤية الهلال بالعين المجردة", "لم تتم رؤية الهلال بالعين المجردة")
    seen_binocular = parse_bool_seen(text, "تمت رؤية الهلال بالمنظار", "لم تتم رؤية الهلال بالمنظار")
    seen_telescope = parse_bool_seen(text, "تمت رؤية الهلال بالمرقب", "لم تتم رؤية الهلال بالمرقب")
    seen_camera = parse_bool_seen(text, "تمت رؤية الهلال باستخدام الكاميرا", "لم تتم رؤية الهلال باستخدام الكاميرا")

    attempted_naked_eye = None
    if "لم يتم تحري الهلال بالعين المجردة" in text:
        attempted_naked_eye = False
    elif "بالعين المجردة" in text:
        attempted_naked_eye = True

    attempted_binocular = None
    if "لم يتم تحري الهلال بالمنظار" in text:
        attempted_binocular = False
    elif "بالمنظار" in text:
        attempted_binocular = True

    attempted_telescope = None
    if "لم يتم تحري الهلال بالمرقب" in text:
        attempted_telescope = False
    elif "بالمرقب" in text:
        attempted_telescope = True

    attempted_camera = None
    if "لم يتم تحري الهلال باستخدام الكاميرا" in text:
        attempted_camera = False
    elif "باستخدام الكاميرا" in text:
        attempted_camera = True

    comment_match = re.search(r"وقال .*?\"(.*)\"", entry_text, flags=re.S)
    comment_text = normalize_space(comment_match.group(1)) if comment_match else None

    return {
        "page_url": page_url,
        "hijri_year": hijri_year,
        "hijri_month_name": hijri_month_name,
        "section_type": section_type,
        "gregorian_date": gregorian_date,
        "country": country,
        "city": city,
        "province": province,
        "obs_time_text": obs_time_text,
        "obs_relative_to_sunset": obs_relative_to_sunset,
        "seen_any": seen_any,
        "seen_naked_eye": seen_naked_eye,
        "seen_binocular": seen_binocular,
        "seen_telescope": seen_telescope,
        "seen_camera": seen_camera,
        "attempted_naked_eye": attempted_naked_eye,
        "attempted_binocular": attempted_binocular,
        "attempted_telescope": attempted_telescope,
        "attempted_camera": attempted_camera,
        "sky_cloud_text": sky_cloud_text,
        "weather_text": weather_text,
        "raw_text": text,
        "comment_text": comment_text,
    }

def parse_month_page(url: str):
    soup = get_soup(url)

    title = soup.find(["h1", "title"])
    title_text = normalize_space(title.get_text(" ", strip=True)) if title else ""

    hijri_year = None
    m_year = re.search(r"(\d{4})\s*هـ", title_text)
    if m_year:
        hijri_year = int(m_year.group(1))

    hijri_month_name = None
    m_month = re.search(r"هلال\s+([^\s]+(?:\s+[^\s]+)?)\s+لعام", title_text)
    if m_month:
        hijri_month_name = normalize_space(m_month.group(1))

    rows = []

    # Find the main content area by starting from the section title
    all_heads = soup.find_all(["h1", "h2", "h3"])
    current_section_type = None
    current_date = None
    current_country = None
    current_country_chunks = []

    def flush_country_block():
        nonlocal rows, current_country_chunks, current_country, current_date, current_section_type
        if not current_country or not current_country_chunks:
            return
        block_text = "\n".join(current_country_chunks).strip()
        entries = split_observation_entries(block_text)
        for e in entries:
            rows.append(parse_entry(
                page_url=url,
                hijri_year=hijri_year,
                hijri_month_name=hijri_month_name,
                section_type=current_section_type or "unknown",
                gregorian_date=current_date,
                country=current_country,
                entry_text=e,
            ))
        current_country_chunks = []

    body = soup.body
    if body is None:
        return rows

    for elem in body.descendants:
        if getattr(elem, "name", None) in ["h1", "h2", "h3"]:
            text = normalize_space(elem.get_text(" ", strip=True))

            if text == "نتائج تحري هلال أول الشهر (رمضان)" or text.startswith("نتائج تحري هلال أول الشهر"):
                flush_country_block()
                current_section_type = "first_crescent"
                current_date = None
                current_country = None
                continue

            if text == "نتائج تحري هلال آخر الشهر (شعبان)" or text.startswith("نتائج تحري هلال آخر الشهر"):
                flush_country_block()
                current_section_type = "last_crescent"
                current_date = None
                current_country = None
                continue

            # date heading like "الثلاثاء 17 فبراير 2026"
            if DATE_RE.search(text):
                flush_country_block()
                current_date = text
                current_country = None
                continue

            # country heading: short heading, not one of the known section titles
            if current_section_type and len(text) < 60 and "نتائج" not in text and not DATE_RE.search(text):
                flush_country_block()
                current_country = text
                continue

        # collect paragraphs/div text under current country
        if current_section_type and current_country:
            if getattr(elem, "name", None) in ["p", "div", "li"]:
                t = normalize_space(elem.get_text(" ", strip=True))
                if "وقت الرصد:" in t:
                    current_country_chunks.append(t)
                elif current_country_chunks and t:
                    current_country_chunks.append(t)

    flush_country_block()
    return rows

def main():
    links = extract_month_links()
    print(f"Found {len(links)} month pages")

    all_rows = []
    for i, link in enumerate(links, 1):
        try:
            print(f"[{i}/{len(links)}] {link['url']}")
            rows = parse_month_page(link["url"])
            all_rows.extend(rows)
            time.sleep(0.7)
        except Exception as e:
            print(f"FAILED {link['url']}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv("crescent_observations_raw.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to crescent_observations_raw.csv")

if __name__ == "__main__":
    main()