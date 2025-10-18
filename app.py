# app_patched.py — Newswire Scanner (Patched)
# Implements hybrid filtering with ">=3 categories" rule.
# Includes: RSS additions, full-article fetch, expanded regex, positive whitelist for clinical improvement lines.

import os, re, time, json, hashlib, requests, feedparser, datetime, threading, random, logging, socket, math
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from openai import OpenAI
from flask import Flask, jsonify
import yfinance as yf
import html as ihtml

# -------------------- ENV --------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GSHEET_KEY = os.getenv("GSHEET_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
REDIS_URL = os.getenv("REDIS_URL")
INSTANCE = socket.gethostname()

# -------------------- APP/LOGGER --------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.logger.setLevel(logging.INFO)
app.logger.info(f"🧭 instance={INSTANCE} | use_redis={'yes' if REDIS_URL else 'no'}")

# -------------------- RSS FEEDS (added per user) --------------------
RSS_FEEDS = [
    # Existing
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States",
    # Added
    "https://www.globenewswire.com/RssFeed/industry/4573-Biotechnology/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Biotechnology",
    "https://www.globenewswire.com/RssFeed/industry/4577-Pharmaceuticals/feedTitle/GlobeNewswire%20-%20Industry%20News%20on%20Pharmaceuticals",
]

# -------------------- HYBRID FILTER SETTINGS --------------------
EMBEDDING_MODEL = "text-embedding-3-small"
SIM_THRESHOLD = 0.82          # keep as-is per user (no #6 needed after #2)
TITLE_BONUS = 0.02            # keep default
MAX_ARTICLE_LEN = 6000

# Negation patterns: we keep standard clinical-negative cues,
# but allow POSITIVE_WHITELIST to override if the sentence indicates clinical improvement.
NEGATIONS = re.compile(r"""(?ix)
    \b(
        did\s+not|does\s+not|do\s+not|
        has\s+not|have\s+not|
        failed\s+to|fail\s+to|
        not\s+statistically\s+significant|
        did\s+not\s+meet|missed
    )\b
""")

# Positive whitelist: clinical improvement phrases (override negation if present in same sentence)
POSITIVE_WHITELIST = re.compile(r"""(?ix)
(
  # Treatment/medication no longer required; tapering; sparing
  no\s+longer\s+(required|need(ed)?)\s*(for)?\s*(treatment|therapy|medication|steroids?|ATD|insulin|oxygen|transfusion|rescue\s+medication)
 |no\s+(rescue|additional|further)\s+(medication|therapy|treatment)\s+required
 |did\s+not\s+require\s+(rescue|additional|further)\s+(medication|therapy|treatment)
 |(weaned|tapered)\s+off\s+(steroids?|therapy|medication)
 |(steroid|opioid)[- ]sparing\s+(effect|benefit)
 |discontinued\s+(steroids?|immunosuppressants?|ATD|therapy)\s+without\s+relapse
 |reduced\s+(dose|use|dependence)\s+of\s+(steroids?|therapy|medication)

  # Remission / drug-free remission / no relapse/exacerbation/progression
 |(treatment|drug)[- ]free\s+remission
 |\bremission(\s+rate|\s+achieved|\s+observed)?\b
 |no\s+(signs|evidence)\s+of\s+disease\b
 |no\s+relapse(s)?\s+observed
 |no\s+exacerbation(s)?\s+reported
 |no\s+disease\s+progression
 |no\s+clinical\s+flare(s)?

  # Normalization / within normal range
 |(normalized|restored\s+to\s+normal)\s+(levels?|function|values?)
 |within\s+(normal|reference)\s+range
 |returned\s+to\s+baseline\s+(levels?|values?)

  # Supportive care not needed
 |no\s+supplemental\s+oxygen\s+required
 |no\s+transfusion\s+required
 |no\s+dialysis\s+required
 |no\s+hospital(ization)?\s+required
 |did\s+not\s+require\s+(hospitalization|dialysis|transfusion|intensive\s+care)

  # Infectious/inflammatory favorable "no"
 |no\s+detectable\s+(virus|viral\s+load|pathogen)
 |no\s+positive\s+(culture|PCR)\s+result
 |no\s+active\s+lesion(s)?
 |no\s+new\s+lesion(s)?

  # Symptoms relief
 |no\s+(significant\s+)?pain\s+reported
 |no\s+symptoms\s+reported
 |did\s+not\s+experience\s+(significant\s+)?pain

  # Misc
 |no\s+dose\s+escalation\s+needed
 |no\s+dose\s+adjustment\s+needed
 |no\s+worsening\s+observed
)
""")

# -------------------- CATEGORY DEFINITIONS --------------------
PHASE2_ONLY = [
    r"end[- ]of[- ]phase 2 meeting (planned|completed|requested|initiated)",
    r"end[- ]of[- ]phase 2 (discussion|meeting) (with|at)\s*fda",
    r"phase 3 (planning|preparation) (initiated|underway)",
    r"advancing to pivotal phase 3 (study|trial)",
    r"design of phase 3 trial (finalized|completed)",
    r"phase 3 protocol (under development|aligned with fda)",
    r"dose[- ]ranging (study|results)",
    r"dose[- ]optimization study (completed|conducted)",
    r"proof[- ]of[- ]concept( (study|trial|results|data))?",
    r"exploratory (endpoints|data) (achieved|support)",
    r"(refining|identified) target patient population",
    r"(optimization of|optimizing) dosing regimen",
    r"bridging study planned (before|prior to) phase 3",
    r"pivotal trial planning (underway|initiated)",
    r"(six|6)[- ]month(s)? (off[- ]treatment|treatment[- ]free) (follow[- ]up|remission)",
]
PHASE2_QUERIES = [
    "end-of-Phase 2 meeting planned with FDA",
    "advancing to a pivotal Phase 3 trial",
    "dose-ranging results inform Phase 3 design",
    "proof-of-concept established enabling late-stage development"
]

PHASE3_ONLY = [
    r"(met|achieved|reached|satisfied)\s+the\s+primary\s+endpoint",
    r"primary (objective|efficacy endpoint) (achieved|met)",
    r"co[- ]primary endpoints (achieved|met)",
    r"(achieved|met|exceeded)\s+(all\s+)?secondary endpoints",
    r"(superiority|non[- ]inferiority)\s+(over|to)\s+(placebo|standard of care|soc|comparator|existing therapies?)",
    r"pre[- ]nda meeting (planned|completed|held)",
    r"\bnda (submission|filing) (expected|initiated|completed|filed|underway)",
    r"rolling (submission|nda) (initiated|ongoing)",
    r"\bbla submission planned\b",
    r"pdufa date (announced|scheduled|set)",
    r"fda (has )?accepted the nda (for review)?",
    r"regulatory (review|submission) (underway|completed)",
    r"(approval decision expected|marketing authorization application|label expansion planned)",
    r"commercial launch preparation (underway|initiated)",
    r"phase 4 post[- ]marketing study planned"
]
PHASE3_QUERIES = [
    "met the primary endpoint in Phase 3",
    "company plans to submit an NDA",
    "FDA accepted the NDA and set a PDUFA date",
    "results will form the basis of regulatory submission"
]

COMMON = [
    r"statistically significant( (improvement|reduction|results))?",
    r"clinically meaningful (improvement|benefit|effect)",
    r"robust (efficacy|clinical response)",
    r"dose[- ]dependent response|dose[- ]response relationship",
    r"(improved|enhanced)\s+(primary outcomes?|response rate|survival rate|pfs|orr|os|qol)",
    r"(durable response|sustained efficacy)",
    r"(consistent efficacy|efficacy maintained)",
    r"no (drug|treatment)[- ]related (serious )?adverse events",
    r"no serious safety signals|no unexpected safety concerns",
    r"well tolerated|favorable tolerability profile|manageable safety profile",
    r"safety profile consistent (with previous (trials|studies)|with prior studies|with standard of care)",
    r"low discontinuation rate due to adverse events",
    r"favorable (risk[- ]benefit|benefit[- ]risk) profile",
    r"(engaging|dialogue|interactions) with (regulators|fda)",
    r"regulatory (alignment|discussions|feedback) (achieved|incorporated|ongoing)",
    r"findings support potential (regulatory )?approval",
    r"data (will|to) form the basis of (an )?(nda|bla|submission)",
    r"(addresses|addressing) an unmet medical need",
    r"(first|best)[- ]in[- ]class (potential|profile)|potential to become standard of care",
]
COMMON_QUERIES = [
    "statistically significant improvement with clinically meaningful benefit",
    "well tolerated with no treatment-related serious adverse events",
    "findings support potential regulatory approval and address unmet medical need"
]



# -------------------- SAFETY DETECTION (for push flag) --------------------
SAFETY_REGEX = re.compile(r"""(?ix)
 (no\s+(drug|treatment)[- ]related\s+(serious\s+)?adverse\s+events?)
 |no\s+serious\s+safety\s+signals?
 |no\s+unexpected\s+safety\s+concerns?
 |well\s+tolerated
 |favorable\s+tolerability\s+profile
 |manageable\s+safety\s+profile
 |safety\s+profile\s+consistent(\s+with\s+previous\s+(trials|studies))?
 |low\s+discontinuation\s+rate\s+due\s+to\s+adverse\s+events?
""")
EFFECT_SIZE = [
    r"clinically meaningful (improvement|benefit|effect)",
    r"(robust|strong|pronounced)\s+efficacy",
    r"(marked|substantial)\s+improvement",
    r"significant effect size|large effect size",
    r"(durable response|sustained efficacy)",
    r"high (response rate|orr|cr|pr)",
    r"deep responses? observed|robust clinical response observed",
    r"meaningful reduction in",
    r"significant magnitude of response|substantial clinical impact",
    r"(treatment[- ]free|drug[- ]free)\s+remission",
    r"\bremission(\s+rate|\s+achieved|\s+observed)?\b",
    r"(normalized|restored\s+to\s+normal)\s+(levels|function)",
    r"within\s+(normal|reference)\s+range",
]
EFFECT_SIZE_QUERIES = [
    "clinically meaningful improvement and robust efficacy",
    "substantial improvement with durable responses",
    "high overall response rate and deep responses observed",
    "treatment-free remission with normalized levels within reference range"
]

COMPARATIVE = [
    r"superior (to|over)\s+(placebo|standard of care|soc|comparator|existing therapies?)",
    r"significantly greater efficacy (vs|than)\s+comparator",
    r"non[- ]inferior (to|versus)\s+\w+",
    r"non[- ]inferior and numerically superior",
    r"outperformed\s+\w+ (in|on)\s+(primary|secondary)\s+endpoints?",
    r"higher response rate (vs|than)\s+(control|comparator)",
    r"favorable efficacy profile compared (with|to) current therapies",
    r"greater magnitude of effect compared (to|with)\s+(baseline|soc|comparator)",
    r"superior benefit observed across endpoints|more effective than existing treatment options",
    r"disease[- ]modifying",
    r"paradigm (change|shifting)",
    r"potential (new )?standard of care",
]
COMPARATIVE_QUERIES = [
    "superior efficacy compared to standard of care",
    "outperformed placebo across primary and secondary endpoints",
    "non-inferior to comparator with higher response rate",
    "disease-modifying potential and a new standard of care"
]


SAFETY_ONLY = [
    r"(no|not)\s+(drug|treatment)[- ]related\s+(serious\s+)?adverse\s+events",
    r"no\s+serious\s+safety\s+signals",
    r"no\s+unexpected\s+safety\s+concerns",
    r"(well|generally)\s+tolerated",
    r"(favorable|acceptable|manageable)\s+(tolerability|safety)\s+profile",
    r"safety(\s+and\s+tolerability)?\s+(profile\s+)?(consistent|aligned|similar|in\s+line)\s+with\s+(prior|previous|earlier|past)\s+(trials?|studies?|data)",
    r"low\s+discontinuation\s+rate\s+due\s+to\s+adverse\s+events",
    r"no\s+dose[- ]limiting\s+toxicities",
    r"safety\s+data\s+support(s)?\s+(progression|continued\s+development|regulatory\s+submission)",
]
SAFETY_QUERIES = [
    "well tolerated across all doses",
    "no treatment-related serious adverse events reported",
    "no unexpected safety concerns identified",
    "safety and tolerability consistent with prior studies",
    "safety profile consistent with previous trials",
    "favorable tolerability profile with low discontinuation due to AEs",
    "no dose-limiting toxicities observed",
    "overall safety profile supports continued development and regulatory submission"
]
CATEGORIES = {
    "safety": {"regex": [re.compile(p, re.I) for p in SAFETY_ONLY], "queries": SAFETY_QUERIES},
    "phase2_only": {"regex": [re.compile(p, re.I) for p in PHASE2_ONLY], "queries": PHASE2_QUERIES},
    "phase3_only": {"regex": [re.compile(p, re.I) for p in PHASE3_ONLY], "queries": PHASE3_QUERIES},
    "common":      {"regex": [re.compile(p, re.I) for p in COMMON],      "queries": COMMON_QUERIES},
    "effect_size": {"regex": [re.compile(p, re.I) for p in EFFECT_SIZE], "queries": EFFECT_SIZE_QUERIES},
    "comparative": {"regex": [re.compile(p, re.I) for p in COMPARATIVE], "queries": COMPARATIVE_QUERIES},
}
REQUIRED_MIN_CATS = 3

# -------------------- DEDUP HELPERS/SECTOR CACHE --------------------
def load_json_dict(path):
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return {}

def save_json_dict(path, d):
    with open(path, "w") as f: json.dump(d, f)

def load_json_set(path):
    if os.path.exists(path):
        with open(path, "r") as f: return set(json.load(f))
    return set()

def save_json_set(path, s):
    with open(path, "w") as f: json.dump(list(s), f)

def hash_uid(url, title):
    return hashlib.sha256((url + "||" + title).encode("utf-8")).hexdigest()[:16]

def clean_url(u: str) -> str:
    p = urlparse(u); return f"{p.scheme}://{p.netloc}{p.path}"

def ny_kst_label(struct_time_or_none):
    if struct_time_or_none:
        dt_utc = datetime.datetime(*struct_time_or_none[:6], tzinfo=ZoneInfo("UTC"))
    else:
        dt_utc = datetime.datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    dt_ny = dt_utc.astimezone(ZoneInfo("America/New_York"))
    dt_kst = dt_utc.astimezone(ZoneInfo("Asia/Seoul"))
    return f"{dt_ny.strftime('%Y-%m-%d %H:%M:%S ET')} ({dt_kst.strftime('%Y-%m-%d %H:%M:%S KST')})"

# -------------------- BODY FETCH / CLEAN --------------------
def fetch_body_text(url: str, timeout=12) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewswireScanner/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            return ""
        html = resp.text or ""
        html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
        text = re.sub(r"(?is)<[^>]+>", " ", html)
        text = ihtml.unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n+", " ", text)
        return text.strip()
    except Exception:
        return ""

# -------------------- TICKER/COMPANY/SECTOR --------------------
TICKER_PATTERNS = [
    r"\b(?:NASDAQ|Nasdaq|NYSE|AMEX|OTC|TSX|ASX|HKEX):\s*([A-Z]{1,5})\b",
    r"\$\b([A-Z]{1,5})\b",
    r"\(([A-Z]{1,5})\)",
    r"\[([A-Z]{1,5})\]",
    r"\bTICKER:\s*([A-Z]{1,5})\b",
    r"\b([A-Z]{2,5})\b"
]
def extract_ticker(text):
    for pat in TICKER_PATTERNS[:-1]:
        m = re.search(pat, text)
        if m: return m.group(1)
    m = re.search(TICKER_PATTERNS[-1], text)
    if m:
        cand = m.group(1)
        if cand.isalpha() and 2 <= len(cand) <= 5:
            return cand
    return None

COMPANY_PATTERNS = [
    r"^(.+?)\s*\((?:NASDAQ|Nasdaq|NYSE|AMEX|OTC|TSX|ASX|HKEX):\s*[A-Z]{1,5}\)\s",
    r"^(.+?)\s+(?:Announces|Launches|Introduces|Signs|Enters|Reports|Unveils|Debuts)\b",
    r"^(.+?)\s+(?:Wins|Receives|Secures|Partners|Collaborates|Expands)\b"
]
def extract_company(text):
    for pat in COMPANY_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            name = m.group(1).strip(" -–—|")
            if 2 <= len(name) <= 80: return name
    return None

def get_sector_with_cache(ticker, cache):
    if not ticker: return "Unknown"
    if ticker in cache and cache[ticker]: return cache[ticker]
    try:
        sector = yf.Ticker(ticker).info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    cache[ticker] = sector
    return sector

# -------------------- OPENAI --------------------
client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_ko(text):
    prompt = (
        "다음 보도자료를 2~3문장으로 한국어로만 요약해줘. "
        "회사명, 제품명, 기관명은 영어 그대로 유지하고 불필요한 해석은 하지 마.\n\nTEXT:\n" + (text or "")[:2000]
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role":"user","content":prompt}],
    )
    return r.choices[0].message.content.strip()

# -------------------- EMBEDDINGS --------------------
EMBED_CACHE_FILE = "embed_cache.json"
_embed_cache = load_json_dict(EMBED_CACHE_FILE)

def _vec(text: str):
    key = f"emb::{hashlib.sha256((text or '').encode('utf-8')).hexdigest()[:16]}"
    if key in _embed_cache: return _embed_cache[key]
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=[text or ""])
    v = r.data[0].embedding
    _embed_cache[key] = v
    if random.random() < 0.05:
        save_json_dict(EMBED_CACHE_FILE, _embed_cache)
    return v

def _cos(u, v):
    num = sum(a*b for a,b in zip(u,v))
    den = math.sqrt(sum(a*a for a in u))*math.sqrt(sum(b*b for b in v))
    return num/den if den else 0.0

CATEGORY_QUERY_EMBEDS = {}
def _warmup_category_queries():
    for name, obj in CATEGORIES.items():
        qemb = []
        for q in obj["queries"]:
            qemb.append(_vec(q))
        CATEGORY_QUERY_EMBEDS[name] = qemb
_warmup_category_queries()

# -------------------- MATCHING --------------------
def _normalize(s: str) -> str:
    return (s or "")[:MAX_ARTICLE_LEN]

def _has_negation(sentence: str) -> bool:
    if POSITIVE_WHITELIST.search(sentence):
        return False
    return bool(NEGATIONS.search(sentence))

def _exact_match_any(regex_list, text):
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    for sent in sentences:
        if _has_negation(sent):
            continue
        for pat in regex_list:
            if pat.search(sent):
                return True
    return False

def _semantic_match(name: str, title: str, body: str) -> bool:
    t = _normalize(title)
    b = _normalize(body)
    tvec = _vec(t) if t else None
    bvec = _vec(b) if b else None
    qvecs = CATEGORY_QUERY_EMBEDS.get(name, [])
    if not qvecs: return False

    def _max_sim(doc_vec):
        if not doc_vec: return 0.0
        sims = [_cos(doc_vec, qv) for qv in qvecs]
        return max(sims) if sims else 0.0

    sim_title = _max_sim(tvec)
    sim_body  = _max_sim(bvec)

    neg_penalty = 0.0
    if NEGATIONS.search(title) or NEGATIONS.search(body):
        if not POSITIVE_WHITELIST.search(title + " " + body):
            neg_penalty = 0.02

    if sim_title + TITLE_BONUS - neg_penalty >= SIM_THRESHOLD:
        return True
    if sim_body - neg_penalty >= SIM_THRESHOLD:
        return True
    return False

def match_categories(title: str, body_text: str):
    text = f"{title} {body_text}".strip()
    matched = []
    dbg = {}
    for name, obj in CATEGORIES.items():
        exact_ok = _exact_match_any(obj["regex"], text)
        sem_ok   = _semantic_match(name, title, body_text)
        ok = exact_ok or sem_ok
        dbg[name] = {"exact": exact_ok, "semantic": sem_ok}
        if ok:
            matched.append(name)
    return matched, dbg

# -------------------- DISCORD --------------------
def push_discord(payload):
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code >= 300:
            app.logger.error(f"[discord] push status {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        app.logger.error(f"[discord] push error: {e}")

def discord_embed(category_labels, title, url, ticker, company, sector, article_time, summary_ko, dbg=None):
    cat_line = " | ".join(category_labels) if category_labels else "unclassified"
    desc = f"{summary_ko}\n\n"
    if dbg:
        bits = [f"{k}:E={'1' if v['exact'] else '0'}/S={'1' if v['semantic'] else '0'}" for k,v in dbg.items()]
        desc += "🧠 match: " + ", ".join(bits) + "\n\n"
    desc += f"🔗 원문: {url}"
    return {
        "username": "Newswire Scanner",
        "embeds": [{
            "title": f"[{cat_line}] {title}",
            "url": url,
            "description": desc,
            "fields": [
                {"name":"Company","value": company or "N/A", "inline": True},
                {"name":"Ticker","value": ticker or "N/A", "inline": True},
                {"name":"Sector","value": sector, "inline": True},
                {"name":"Article Time","value": article_time, "inline": False},
            ],
            "footer": {"text":"Source: Multiple RSS"}
        }]
    }

# -------------------- REDIS DEDUP --------------------
try:
    import redis
except Exception:
    redis = None

_rds = None
def _get_redis():
    global _rds
    if _rds is None and REDIS_URL and redis:
        try:
            _rds = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                ssl=True if REDIS_URL.startswith("rediss://") else None
            )
            app.logger.info("[dedup] Redis connected")
        except Exception as e:
            app.logger.error(f"[dedup] Redis connect failed: {e}")
            _rds = None
    return _rds

def seen_check_and_set_url(guid: str, ttl_seconds: int = 14*24*3600) -> bool:
    r = _get_redis()
    if r:
        key = f"nwscanner:seen:url:{guid}"
        try:
            ok = r.setnx(key, INSTANCE)
            if ok: r.expire(key, ttl_seconds)
            return not ok
        except Exception as e:
            app.logger.error(f"[dedup] Redis error(url): {e}")
    seen = load_json_set("seen.json")
    if guid in seen: return True
    seen.add(guid); save_json_set("seen.json", seen)
    return False

def _normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def seen_set_title_once(title: str, ttl_seconds: int = 14*24*3600) -> bool:
    norm = _normalize_title(title)
    if not norm: return False
    r = _get_redis()
    if r:
        key = f"nwscanner:seen:title:{hashlib.sha256(norm.encode()).hexdigest()[:16]}"
        try:
            ok = r.setnx(key, INSTANCE)
            if ok: r.expire(key, ttl_seconds)
            return not ok
        except Exception as e:
            app.logger.error(f"[dedup] Redis error(title): {e}")
            return False
    return False

# -------------------- SCAN LOOP --------------------
def run_once():
    app.logger.info("🔎 run_once: start")
    sector_cache = load_json_dict("sector_cache.json")
    now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    for feed_url in RSS_FEEDS:
        app.logger.info(f"📰 fetching feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        app.logger.info(f"📰 entries: {len(feed.entries)}")

        for e in feed.entries[:120]:
            url = getattr(e, "link", "") or ""
            if not url:
                app.logger.info("⚠️  entry skipped: no URL")
                continue
            guid = hashlib.sha256(url.encode()).hexdigest()[:16]

            if seen_check_and_set_url(guid):
                app.logger.info(f"⏩ already seen(url): {guid}")
                continue

            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""

            # Fetch full body and use for matching
            body_text = fetch_body_text(url)
            text_for_match = body_text if body_text and len(body_text) > 200 else summary

            matched_labels, dbg = match_categories(title, text_for_match)
            if len(matched_labels) < 3:
                app.logger.info(f"❌ <3 categories: {title[:100]}...")
                continue

            if seen_set_title_once(title):
                app.logger.info(f"⏩ already seen(title): {title[:80]}…")
                continue

            app.logger.info(f"✅ MATCH({len(matched_labels)} cats): {title[:120]}... — {matched_labels}")

            article_time = ny_kst_label(
                getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            )
            hay = f"{title} {summary} {text_for_match}"
            ticker = extract_ticker(hay)
            company = extract_company(title) or extract_company(summary)
            sector = get_sector_with_cache(ticker, sector_cache)

            try:
                ko = summarize_ko(text_for_match if text_for_match else title)
            except Exception as ex:
                ko = "(요약 실패) " + (summary[:200] or title)
                app.logger.error(f"[gpt] error: {ex}")

            # --- [PATCH START] Enhanced payload with matched categories & E/S hits + Safety flag ---

            # Build readable E/S + hit list per category
            match_lines = []
            for k, v in dbg.items():
                e = '1' if v['exact'] else '0'
                s = '1' if v['semantic'] else '0'
                hit = '✅' if k in matched_labels else '❌'
                match_lines.append(f"{k}:E={e}/S={s}{hit}")

            # Human-friendly category line and bulleted list
            cat_line = " | ".join(matched_labels) if matched_labels else "unclassified"
            matched_list_bullets = "• " + "\n• ".join(matched_labels) if matched_labels else "• (none)"

            # Safety mention detection from available article text
            hay_for_safety = (text_for_match or "") + " " + (title or "") + " " + (summary or "")
            has_safety = ("safety" in matched_labels) or bool(SAFETY_REGEX.search(hay_for_safety))
            safety_line = "🩺 Safety Mention: " + ("✅ 포함됨" if has_safety else "❌ 언급 없음")

            payload = {
                "username": "Newswire Scanner",
                "embeds": [{
                    "title": f"[{cat_line}] {title}",
                    "url": url,
                    "description": (
                        f"🧩 Matched Categories: {cat_line}\n\n"
                        f"{ko}\n\n"
                        f"{safety_line}\n\n"
                        "🧠 match: " + ", ".join(match_lines) +
                        f"\n\n🔗 원문: {url}"
                    ),
                    "fields": [
                        {"name":"Company","value": company or "N/A", "inline": True},
                        {"name":"Ticker","value": ticker or "N/A", "inline": True},
                        {"name":"Sector","value": sector, "inline": True},
                        {"name":"Matched Categories (detail)","value": matched_list_bullets, "inline": False},
                        {"name":"Safety Mention","value": "✅ 있음" if has_safety else "❌ 없음", "inline": True},
                        {"name":"Article Time","value": article_time, "inline": False},
                    ],
                    "footer": {"text":"Source: Multiple RSS"}
                }]
            }
            push_discord(payload)

            # --- [PATCH END] ---
            row = [
                now_kst, feed_url.split('/')[2], guid,
                ticker or "", company or "", sector, "|".join(matched_labels),
                title, article_time, ko, url
            ]
            # append to sheets only if configured externally; stubbed out here

    save_json_dict("embed_cache.json", _embed_cache)
    save_json_dict("sector_cache.json", sector_cache)
    app.logger.info("🔎 run_once: done")

stop_event = threading.Event()
def scanner_loop():
    app.logger.info("🚀 scanner_loop: started")
    while not stop_event.is_set():
        try:
            run_once()
        except Exception as e:
            app.logger.error(f"[scanner] error: {e}")
        delay = random.uniform(35, 50)
        app.logger.info(f"⏱️ sleeping {delay:.1f}s")
        time.sleep(delay)

start_time = time.time()

@app.route("/")
def root():
    return "OK", 200

@app.route("/healthz")
def healthz():
    return jsonify({"status":"ok","uptime_sec": int(time.time()-start_time)}), 200

def main():
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT", "8000"))
    app.logger.info(f"🌐 starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
