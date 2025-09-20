#!/usr/bin/env python3
"""
Detect "famous" depicted people in paintings using Wikidata + Wikipedia signals.
(Enhanced: LOCATION hinting, author normalization, multi-language pageviews)

Pipeline (per row: TITLE, AUTHOR, LOCATION):
1) Find the painting on Wikidata (best-effort) using its TITLE (+ AUTHOR normalization + LOCATION hint).
2) From the painting item, read P180 (depicts). Keep only humans (instance of Q5).
3) For each depicted person:
   - Get Wikipedia sitelinks (en,it,de,fr,nl,es) titles (if any).
   - Get Wikidata sitelink count (across all languages).
   - Fetch last-year pageviews for those language projects via REST API and sum.
4) Classify as "famous" if (pageviews_last_365_sum >= pv_year_threshold) OR (sitelinks >= sitelinks_threshold).

Outputs a tall CSV: one row per (painting, person).

Requirements:
- Python 3.9+
- pip install requests

Usage:
  python detect_famous_from_paintings.py \
    --in data_for_custom_search.csv \
    --out fame_results.csv \
    --pv-year-threshold 12000 \
    --sitelinks-threshold 30 \
    --limit-rows 0 \
    --sleep 0.5

Notes:
- TITLE/AUTHOR/LOCATION マッチングは厳密でないため、誤マッチの可能性あり。
  出力の painting_qid / person_qid / person_label を確認してください。
"""

import csv
import sys
import time
import math
import json
import argparse
import datetime as dt
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import requests

WD_SPARQL = "https://query.wikidata.org/sparql"
WD_API    = "https://www.wikidata.org/w/api.php"
UA        = "FameDetector/1.1 (research; contact: you@example.com)"  # ←適宜編集

LANGS = ["en", "it", "de", "fr", "nl", "es"]

# --------- Helpers ---------
def norm_space(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().split())

def strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def german_unfold(s: str) -> str:
    # Common unfoldings
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = s.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    return s

def normalize_name(s: str) -> str:
    s = norm_space(s)
    s = german_unfold(s)
    s = strip_diacritics(s)
    s = s.lower()
    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in re.split(r"\W+", s) if t]

def token_overlap(a: str, b: str) -> int:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb)

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def backoff_sleep(base: float, attempt: int, cap: float = 8.0):
    time.sleep(min(cap, base * (2 ** attempt)) + 0.05 * math.sin(time.time()))

def request_json(url: str, *, method: str = "GET", params=None, headers=None, data=None, timeout=30) -> Any:
    headers = {"User-Agent": UA, **(headers or {})}
    for attempt in range(5):
        try:
            if method == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, params=params, headers=headers, data=data, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(0.6, attempt)
                continue
            try:
                return r.json()
            except Exception:
                return {"error": f"HTTP {r.status_code}", "text": r.text[:300]}
        except requests.RequestException as e:
            backoff_sleep(0.6, attempt)
    return {"error": "exhausted retries"}

def sparql(query: str) -> Dict[str, Any]:
    params = {"query": query, "format": "json"}
    return request_json(WD_SPARQL, params=params, headers={"Accept": "application/sparql-results+json"})

# --------- Core logic ---------
def guess_painting_qid_by_title_author_location(title: str, author: str, location_hint: str) -> Optional[str]:
    """
    Find a painting QID by title with scoring using AUTHOR normalization and LOCATION hint.
    1) wbsearchentities by title (top 10)
    2) SPARQL verify: must be painting (P31=Q3305213). Fetch creator/collection/location/admin/country labels.
    3) Score:
       +3 if normalized creator label matches author (contains or token overlap>=2)
       +2 if any of collection/location/admin/country label contains location hint (normalized, token overlap>=1)
       +1 otherwise (fallback if nothing matches)
    Return best-scored QID.
    """
    title = norm_space(title)
    author_n = normalize_name(author)
    loc_n = normalize_name(location_hint)

    params = {
        "action": "wbsearchentities",
        "search": title,
        "language": "en",
        "type": "item",
        "limit": 10,
        "format": "json"
    }
    data = request_json(WD_API, params=params)
    candidates = []
    for e in (data.get("search") or []):
        qid = e.get("id")
        if qid and qid.startswith("Q"):
            candidates.append(qid)
    if not candidates:
        return None

    values = " ".join(f"(wd:{qid})" for qid in candidates)
    langlist = ",".join(LANGS)
    q = f"""
    SELECT ?item ?creatorLabel ?collectionLabel ?locationLabel ?adminLabel ?countryLabel WHERE {{
      VALUES (?item) {{ {values} }}
      ?item wdt:P31 wd:Q3305213 .
      OPTIONAL {{ ?item wdt:P170 ?creator . }}
      OPTIONAL {{ ?item wdt:P195 ?collection . }}
      OPTIONAL {{ ?item wdt:P276 ?location . }}
      OPTIONAL {{ ?item wdt:P131 ?admin . }}
      OPTIONAL {{ ?item wdt:P17  ?country . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langlist}" . }}
    }}
    """
    res = sparql(q)
    rows = res.get("results", {}).get("bindings", [])
    if not rows:
        return None

    scored: List[Tuple[int, str]] = []
    for b in rows:
        item_uri = b["item"]["value"]
        qid = item_uri.split("/")[-1]
        creator_label = normalize_name(b.get("creatorLabel", {}).get("value", ""))
        collection_label = normalize_name(b.get("collectionLabel", {}).get("value", ""))
        location_label = normalize_name(b.get("locationLabel", {}).get("value", ""))
        admin_label = normalize_name(b.get("adminLabel", {}).get("value", ""))
        country_label = normalize_name(b.get("countryLabel", {}).get("value", ""))

        score = 0
        # Author match
        if author_n:
            if author_n in creator_label or token_overlap(author_n, creator_label) >= 2:
                score += 3
        else:
            score += 1

        # Location hint match (any of the location-ish labels)
        if loc_n:
            loc_labels = " ".join([collection_label, location_label, admin_label, country_label])
            if loc_n in loc_labels or token_overlap(loc_n, loc_labels) >= 1:
                score += 2

        # light fallback
        if score == 0:
            score = 1

        scored.append((score, qid))

    scored.sort(reverse=True)  # by score desc, then qid desc (stable enough)
    return scored[0][1] if scored else None

def get_depicted_humans_with_signals(painting_qid: str, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Return list of depicted humans with labels, per-language wiki titles, sitelinks count,
    and aggregated Wikipedia language list & count.

    強化点：
      - (wdt:P180 | wdt:P921) で人物を拾う
      - 0件なら人物側の wdt:P1299 で逆引き
      - 人物(Q5)のみに限定（FILTER EXISTS）
    """
    # 言語別Wikipediaタイトルの OPTIONAL ブロック
    opt_lang_blocks = []
    for lang in LANGS:
        site = f"https://{lang}.wikipedia.org/"
        opt_lang_blocks.append(f"""
          OPTIONAL {{
            ?{lang}Sitelink schema:about ?depicted ;
                            schema:isPartOf <{site}> ;
                            schema:name ?{lang}Title .
          }}
        """)
    opt_lang_text = "\n".join(opt_lang_blocks)
    langlist = ",".join(LANGS)

    def _run_union_query() -> List[Dict[str, Any]]:
        q = f"""
        SELECT ?depicted ?depictedLabel ?sitelinks {' '.join('?' + l + 'Title' for l in LANGS)}
               (GROUP_CONCAT(DISTINCT ?wpLang;separator="|") AS ?wpLangs)
               (COUNT(DISTINCT ?wpLang) AS ?wpLangsCount)
        WHERE {{
          wd:{painting_qid} (wdt:P180|wdt:P921) ?depicted .
          FILTER EXISTS {{ ?depicted wdt:P31 wd:Q5 }}
          OPTIONAL {{ ?depicted wikibase:sitelinks ?sitelinks . }}
          {opt_lang_text}
          OPTIONAL {{
            ?anySitelink schema:about ?depicted ;
                         schema:isPartOf ?wiki .
            FILTER(CONTAINS(STR(?wiki), "wikipedia.org"))
            BIND(REPLACE(STR(?wiki), "^https?://([a-zA-Z-]+)\\.wikipedia\\.org/.*$", "$1") AS ?wpLang)
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langlist}" . }}
        }}
        GROUP BY ?depicted ?depictedLabel ?sitelinks {' '.join('?' + l + 'Title' for l in LANGS)}
        """
        res = sparql(q)
        # SPARQLエラーの可視化（任意）
        if debug and 'error' in res:
            print("[DEBUG] SPARQL error (union):", res.get('error'))
        return res.get("results", {}).get("bindings", [])

    def _run_fallback_p1299() -> List[Dict[str, Any]]:
        q = f"""
        SELECT ?depicted ?depictedLabel ?sitelinks {' '.join('?' + l + 'Title' for l in LANGS)}
               (GROUP_CONCAT(DISTINCT ?wpLang;separator="|") AS ?wpLangs)
               (COUNT(DISTINCT ?wpLang) AS ?wpLangsCount)
        WHERE {{
          ?depicted wdt:P1299 wd:{painting_qid} .
          FILTER EXISTS {{ ?depicted wdt:P31 wd:Q5 }}
          OPTIONAL {{ ?depicted wikibase:sitelinks ?sitelinks . }}
          {opt_lang_text}
          OPTIONAL {{
            ?anySitelink schema:about ?depicted ;
                         schema:isPartOf ?wiki .
            FILTER(CONTAINS(STR(?wiki), "wikipedia.org"))
            BIND(REPLACE(STR(?wiki), "^https?://([a-zA-Z-]+)\\.wikipedia\\.org/.*$", "$1") AS ?wpLang)
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langlist}" . }}
        }}
        GROUP BY ?depicted ?depictedLabel ?sitelinks {' '.join('?' + l + 'Title' for l in LANGS)}
        """
        res = sparql(q)
        if debug and 'error' in res:
            print("[DEBUG] SPARQL error (p1299):", res.get('error'))
        return res.get("results", {}).get("bindings", [])

    rows = _run_union_query()
    if debug:
        print(f"[DEBUG] union rows: {len(rows)} for {painting_qid}")
    if not rows:
        rows = _run_fallback_p1299()
        if debug:
            print(f"[DEBUG] fallback p1299 rows: {len(rows)} for {painting_qid}")

    out = []
    for b in rows:
        person_uri = b["depicted"]["value"]
        qid = person_uri.split("/")[-1]
        label = b.get("depictedLabel", {}).get("value", "")

        sitelinks = b.get("sitelinks", {}).get("value")
        try:
            sitelinks = int(sitelinks) if sitelinks is not None else None
        except Exception:
            sitelinks = None

        wp_langs_raw = b.get("wpLangs", {}).get("value", "")
        wp_langs_list = [x for x in wp_langs_raw.split("|") if x]
        wp_langs_count = b.get("wpLangsCount", {}).get("value")
        try:
            wp_langs_count = int(wp_langs_count) if wp_langs_count is not None else None
        except Exception:
            wp_langs_count = None

        titles = {}
        for l in LANGS:
            key = f"{l}Title"
            titles[l] = b.get(key, {}).get("value", "")

        out.append({
            "person_qid": qid,
            "person_label": label,
            "titles": titles,
            "sitelinks": sitelinks,
            "wp_langs": wp_langs_list,
            "wp_langs_count": wp_langs_count,
        })
    return out



def wikipedia_pageviews_last_year(project: str, title: str) -> Optional[int]:
    """
    Sum pageviews for the last 365 days for the given Wikipedia article.
    project like 'en.wikipedia.org', title already URL-safe-ish (space -> _).
    """
    if not title:
        return None
    title = title.replace(" ", "_")
    end = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y%m%d")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{title}/daily/{start}/{end}"
    data = request_json(url)
    if "items" not in data:
        return None
    return sum(item.get("views", 0) for item in data["items"])

def classify_famous(pageviews_sum: Optional[int], sitelinks: Optional[int],
                    pv_year_threshold: int, sitelinks_threshold: int) -> Tuple[bool, str]:
    reasons = []
    is_famous = False
    if pageviews_sum is not None and pageviews_sum >= pv_year_threshold:
        is_famous = True
        reasons.append(f"pv_sum_365={pageviews_sum}>={pv_year_threshold}")
    if sitelinks is not None and sitelinks >= sitelinks_threshold:
        is_famous = True
        reasons.append(f"sitelinks={sitelinks}>={sitelinks_threshold}")
    return is_famous, "; ".join(reasons)

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="CSV with TITLE, AUTHOR, LOCATION")
    ap.add_argument("--out", dest="output_csv", required=True, help="Output CSV (tall format)")
    ap.add_argument("--limit-rows", type=int, default=0, help="Limit number of input rows (0 = no limit)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between items (politeness)")
    ap.add_argument("--pv-year-threshold", type=int, default=12000, help="Famous if SUM(pageviews last 365d across langs) >= this")
    ap.add_argument("--sitelinks-threshold", type=int, default=30, help="Famous if sitelinks >= this")
    args = ap.parse_args()

    # Prepare output
    out_exists = False
    try:
        out_exists = open(args.output_csv, "r", encoding="utf-8")
        out_exists.close()
        out_exists = True
    except Exception:
        out_exists = False

    out_f = open(args.output_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if not out_exists:
        header = [
            "TITLE","AUTHOR","LOCATION","painting_qid",
            "person_qid","person_label",
        ]
        # per-language pageviews
        for lang in LANGS:
            header.append(f"pv_{lang}")
        header += [
            "pv_sum_365","sitelinks","wp_langs_count","wp_langs","is_famous","reasons","timestamp_utc"
        ]
        writer.writerow(header)

    processed = 0
    with open(args.input_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit_rows and processed >= args.limit_rows:
                break
            title = norm_space(row.get("TITLE"))
            author = norm_space(row.get("AUTHOR"))
            location = norm_space(row.get("LOCATION"))
            if not title:
                continue

            painting_qid = guess_painting_qid_by_title_author_location(title, author, location)

            if not painting_qid:
                # couldn't find painting
                row_out = [title, author, location, "", "", ""]
                for _ in LANGS:
                    row_out.append("")
                row_out += ["", "", "FALSE", "painting not found", now_utc_iso()]
                writer.writerow(row_out)
                out_f.flush()
                processed += 1
                time.sleep(args.sleep)
                continue

            # Get depicted persons
            people = get_depicted_humans_with_signals(painting_qid)
            if not people:
                row_out = [title, author, location, painting_qid, "", ""]
                for _ in LANGS:
                    row_out.append("")
                row_out += ["", "", "FALSE", "no depicted humans", now_utc_iso()]
                writer.writerow(row_out)
                out_f.flush()
                processed += 1
                time.sleep(args.sleep)
                continue

            # For each person, get pageviews per language and classify
            for p in people:
                pv_langs = {}
                pv_sum = 0
                for lang in LANGS:
                    proj = f"{lang}.wikipedia.org"
                    t = p["titles"].get(lang, "")
                    pv = wikipedia_pageviews_last_year(proj, t) if t else None
                    pv_langs[lang] = pv
                    if pv is not None:
                        pv_sum += pv

                sl = p.get("sitelinks")
                is_famous, reasons = classify_famous(pv_sum, sl, args.pv_year_threshold, args.sitelinks_threshold)

                row_out = [
                    title, author, location, painting_qid,
                    p.get("person_qid",""), p.get("person_label","")
                ]
                for lang in LANGS:
                    row_out.append("" if pv_langs[lang] is None else pv_langs[lang])
                row_out += [
                    pv_sum if pv_sum else "",
                    sl if sl is not None else "",
                    p.get("wp_langs_count", ""),
                    ",".join(p.get("wp_langs", []) or []),
                    "TRUE" if is_famous else "FALSE",
                    reasons or "",
                    now_utc_iso()
                ]
                writer.writerow(row_out)
                out_f.flush()

            processed += 1
            time.sleep(args.sleep)

    out_f.close()
    print(f"Done. Processed {processed} painting rows.")

if __name__ == "__main__":
    main()
