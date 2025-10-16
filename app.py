# app.py — Newswire Scanner (PRNewswire + GlobeNewswire)
# 하이브리드 키워드 필터(정확매칭 + 의미유사도) & "5대 카테고리 중 ≥3개 충족" 규칙 적용 버전
# - RSS 수집
# - 카테고리별 매칭(Phase2 전용 / Phase3 전용 / 공통 / 효과크기 / 비교우위성)
# - 부정문(negation) 무효화
# - 제목 가중치
# - OpenAI 임베딩 사전계산 + 캐시
# - 기존 기능(요약/Discord/GSheet/중복/헬스체크) 유지

import os, re, time, json, hashlib, requests, feedparser, datetime, threading, random, logging, socket
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from openai import OpenAI
from flask import Flask, jsonify
import yfinance as yf
import math

# -------------------- 환경 변수 --------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GSHEET_KEY = os.getenv("GSHEET_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
REDIS_URL = os.getenv("REDIS_URL")  # ex) rediss://:password@host:port (없으면 로컬 폴백)
INSTANCE = socket.gethostname()

# -------------------- 앱/로거 --------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app.logger.setLevel(logging.INFO)
app.logger.info(f"🧭 instance={INSTANCE} | use_redis={'yes' if REDIS_URL else 'no'}")

# -------------------- RSS 소스 --------------------
RSS_FEEDS = [
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States"
]

# -------------------- 하이브리드 필터 설정 --------------------
# 임베딩 모델/임계값 등
EMBEDDING_MODEL = "text-embedding-3-small"  # 속도/비용 최적
SIM_THRESHOLD = 0.82                         # 의미 유사도 임계값(카테고리 대표문장 vs 기사)
TITLE_BONUS = 0.02                           # 제목 가중: 타이트한 임계값 경계에서 밀어줌
MAX_ARTICLE_LEN = 6000                       # 임베딩용 텍스트 최대 길이

# 부정(negation) 패턴: 해당 문장에 있으면 긍정 키워드 무효화
NEGATIONS = re.compile(r"""(?ix)
    \b(did\s+not|does\s+not|do\s+not|has\s+not|have\s+not|failed\s+to|fail\s+to|no\s+evidence\s+of|
       not\s+statistically\s+significant|did\s+not\s+meet|missed|without\s+meaningful\s+improvement)\b
""")

# 5개 대카테고리 정의(정확 매칭용 키워드 + 임베딩용 대표 문장 세트)
PHASE2_ONLY = [
    # 핵심 키워드/동의어/파생
    r"end[- ]of[- ]phase 2 meeting (planned|completed|requested|initiated)",
    r"end[- ]of[- ]phase 2 (discussion|meeting) (with|at)\s*fda",
    r"phase 3 (planning|preparation) (initiated|underway)",
    r"advancing to pivotal phase 3 (study|trial)",
    r"design of phase 3 trial (finalized|completed)",
    r"phase 3 protocol (under development|aligned with fda)",
    r"dose[- ]ranging (study|results)",
    r"dose[- ]optimization study (completed|conducted)",
    r"proof[- ]of[- ]concept (established|demonstrated|confirmed)",
    r"exploratory (endpoints|data) (achieved|support)",
    r"(refining|identified) target patient population",
    r"(optimization of|optimizing) dosing regimen",
    r"bridging study planned (before|prior to) phase 3",
    r"pivotal trial planning (underway|initiated)",
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
    # 효능
    r"statistically significant( (improvement|reduction|results))?",
    r"clinically meaningful (improvement|benefit|effect)",
    r"robust (efficacy|clinical response)",
    r"dose[- ]dependent response|dose[- ]response relationship",
    r"(improved|enhanced)\s+(primary outcomes?|response rate|survival rate|pfs|orr|os|qol)",
    r"(durable response|sustained efficacy)",
    r"(consistent efficacy|efficacy maintained)",
    # 안전성
    r"no (drug|treatment)[- ]related (serious )?adverse events",
    r"no serious safety signals|no unexpected safety concerns",
    r"well tolerated|favorable tolerability profile|manageable safety profile",
    r"safety profile consistent (with previous (trials|studies)|with prior studies|with standard of care)",
    r"low discontinuation rate due to adverse events",
    r"favorable (risk[- ]benefit|benefit[- ]risk) profile",
    # 규제/전략
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
]

EFFECT_SIZE_QUERIES = [
    "clinically meaningful improvement and robust efficacy",
    "substantial improvement with durable responses",
    "high overall response rate and deep responses observed"
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
    r"superior benefit observed across endpoints|more effective than existing treatment options"
]

COMPARATIVE_QUERIES = [
    "superior efficacy compared to standard of care",
    "outperformed placebo across primary and secondary endpoints",
    "non-inferior to comparator with higher response rate"
]

# 카테고리 컨테이너
CATEGORIES = {
    "phase2_only": {"regex": [re.compile(p, re.I) for p in PHASE2_ONLY], "queries": PHASE2_QUERIES},
    "phase3_only": {"regex": [re.compile(p, re.I) for p in PHASE3_ONLY], "queries": PHASE3_QUERIES},
    "common":      {"regex": [re.compile(p, re.I) for p in COMMON],      "queries": COMMON_QUERIES},
    "effect_size": {"regex": [re.compile(p, re.I) for p in EFFECT_SIZE], "queries": EFFECT_SIZE_QUERIES},
    "comparative": {"regex": [re.compile(p, re.I) for p in COMPARATIVE], "queries": COMPARATIVE_QUERIES},
}

REQUIRED_MIN_CATS = 3  # ≥3개 카테고리 충족해야 통과

# -------------------- 파일 경로/지터 --------------------
SEEN_FILE = "seen.json"
SECTOR_CACHE_FILE = "sector_cache.json"
EMBED_CACHE_FILE = "embed_cache.json"
JITTER_MIN, JITTER_MAX = 35, 50

# -------------------- 선택적: Google Sheets --------------------
use_sheets = bool(GSHEET_KEY and GOOGLE_SERVICE_ACCOUNT_JSON)

def _load_sa_json(raw: str):
    import base64, json, os
    if not raw:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is empty")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict): return obj
        if isinstance(obj, str): return json.loads(obj)
    except Exception:
        pass
    try:
        dec = base64.b64decode(raw).decode("utf-8")
        obj = json.loads(dec)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    if os.path.isfile(raw):
        with open(raw, "r") as f:
            return json.load(f)
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON must be a JSON string, base64-encoded JSON, or a file path")

if use_sheets:
    import gspread
    from google.oauth2.service_account import Credentials

    def append_sheet(row):
        creds = Credentials.from_service_account_info(
            _load_sa_json(GOOGLE_SERVICE_ACCOUNT_JSON),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEET_KEY)
        try:
            ws = sh.worksheet("news_log")
        except Exception:
            ws = sh.add_worksheet(title="news_log", rows=100, cols=12)
        ws.append_row(row, value_input_option="USER_ENTERED")
else:
    def append_sheet(row):
        return

# -------------------- 유틸 --------------------
def load_json_set(path):
    if os.path.exists(path):
        with open(path, "r") as f: return set(json.load(f))
    return set()

def save_json_set(path, s):
    with open(path, "w") as f: json.dump(list(s), f)

def load_json_dict(path):
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return {}

def save_json_dict(path, d):
    with open(path, "w") as f: json.dump(d, f)

def hash_uid(url, title):
    return hashlib.sha256((url + "||" + title).encode("utf-8")).hexdigest()[:16]

def clean_url(u: str) -> str:
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}{p.path}"

def ny_kst_label(struct_time_or_none):
    if struct_time_or_none:
        dt_utc = datetime.datetime(*struct_time_or_none[:6], tzinfo=ZoneInfo("UTC"))
    else:
        dt_utc = datetime.datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    dt_ny = dt_utc.astimezone(ZoneInfo("America/New_York"))
    dt_kst = dt_utc.astimezone(ZoneInfo("Asia/Seoul"))
    return f"{dt_ny.strftime('%Y-%m-%d %H:%M:%S ET')} ({dt_kst.strftime('%Y-%m-%d %H:%M:%S KST')})"

# -------------------- 티커/회사/섹터 --------------------
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

# -------------------- OpenAI 클라이언트/요약 --------------------
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

# -------------------- 임베딩(사전 계산 + 캐시) --------------------
_embed_cache = load_json_dict(EMBED_CACHE_FILE)

def _vec(text: str):
    key = f"emb::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
    if key in _embed_cache: return _embed_cache[key]
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    v = r.data[0].embedding
    _embed_cache[key] = v
    # 주기적으로 저장
    if random.random() < 0.05:
        save_json_dict(EMBED_CACHE_FILE, _embed_cache)
    return v

def _cos(u, v):
    # 두 벡터는 동일 모델 차원이라고 가정
    num = sum(a*b for a,b in zip(u,v))
    den = math.sqrt(sum(a*a for a in u))*math.sqrt(sum(b*b for b in v))
    return num/den if den else 0.0

# 카테고리 대표 쿼리 임베딩 사전 계산
CATEGORY_QUERY_EMBEDS = {}
def _warmup_category_queries():
    for name, obj in CATEGORIES.items():
        qemb = []
        for q in obj["queries"]:
            qemb.append(_vec(q))
        CATEGORY_QUERY_EMBEDS[name] = qemb

_warmup_category_queries()

# -------------------- 하이브리드 매칭 로직 --------------------
def _normalize(s: str) -> str:
    s = (s or "")
    # 임베딩 입력 길이 제한
    return s[:MAX_ARTICLE_LEN]

def _has_negation(sentence: str) -> bool:
    return bool(NEGATIONS.search(sentence))

def _exact_match_any(regex_list, text):
    # 부정문 단위 처리를 위해 문장별 검사
    # 간단한 문장 분할(마침표/느낌표/물음표 기준)
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    for sent in sentences:
        if _has_negation(sent):
            continue
        for pat in regex_list:
            if pat.search(sent):
                return True
    return False

def _semantic_match(name: str, title: str, body: str) -> bool:
    # 제목/본문 임베딩 한 번씩
    t = _normalize(title)
    b = _normalize(body)
    tvec = _vec(t) if t else None
    bvec = _vec(b) if b else None
    qvecs = CATEGORY_QUERY_EMBEDS.get(name, [])
    if not qvecs: return False

    # 제목 가중치: 제목 유사도에 TITLE_BONUS 추가
    def _max_sim(doc_vec):
        if not doc_vec: return 0.0
        sims = [_cos(doc_vec, qv) for qv in qvecs]
        return max(sims) if sims else 0.0

    sim_title = _max_sim(tvec)
    sim_body  = _max_sim(bvec)

    # 부정문이 전체 텍스트에 강하게 깔리면 임계 상향
    neg_penalty = 0.0
    if NEGATIONS.search(title) or NEGATIONS.search(body):
        neg_penalty = 0.02

    # 임계 판단
    if sim_title + TITLE_BONUS - neg_penalty >= SIM_THRESHOLD:
        return True
    if sim_body - neg_penalty >= SIM_THRESHOLD:
        return True
    return False

def match_categories(title: str, summary: str):
    """
    반환: (matched_labels: list[str], debug_scores: dict)
    - 각 카테고리는 '정확매칭 OR 의미매칭'이면 충족
    - 부정문은 정확매칭 단계에서 무효화
    """
    text = f"{title} {summary}".strip()
    matched = []
    dbg = {}
    for name, obj in CATEGORIES.items():
        exact_ok = _exact_match_any(obj["regex"], text)
        sem_ok   = _semantic_match(name, title, summary)
        ok = exact_ok or sem_ok
        dbg[name] = {"exact": exact_ok, "semantic": sem_ok}
        if ok:
            matched.append(name)
    return matched, dbg

# -------------------- Discord --------------------
def discord_embed(category_labels, title, url, ticker, company, sector, article_time, summary_ko, dbg=None):
    cat_line = " | ".join(category_labels) if category_labels else "unclassified"
    desc = f"{summary_ko}\n\n"
    if dbg:
        try:
            # 간단 디버그(선택): 어떤 방식으로 매칭됐는지 표시
            bits = []
            for k,v in dbg.items():
                bits.append(f"{k}:E={'1' if v['exact'] else '0'}/S={'1' if v['semantic'] else '0'}")
            desc += "🧠 match: " + ", ".join(bits) + "\n\n"
        except Exception:
            pass
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
            "footer": {"text":"Source: PRNewswire / GlobeNewswire"}
        }]
    }

def push_discord(payload):
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code >= 300:
            app.logger.error(f"[discord] push status {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        app.logger.error(f"[discord] push error: {e}")

# -------------------- Dedup Store (Redis or local) --------------------
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

def _normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

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
    seen = load_json_set(SEEN_FILE)
    if guid in seen: return True
    seen.add(guid); save_json_set(SEEN_FILE, seen)
    return False

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

# -------------------- 스캔 1회 --------------------
def run_once():
    app.logger.info("🔎 run_once: start")
    sector_cache = load_json_dict(SECTOR_CACHE_FILE)
    now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    for feed_url in RSS_FEEDS:
        app.logger.info(f"📰 fetching feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        app.logger.info(f"📰 entries: {len(feed.entries)}")

        for e in feed.entries[:80]:
            url = getattr(e, "link", "") or ""
            if not url:
                app.logger.info("⚠️  entry skipped: no URL")
                continue

            clean = clean_url(url)
            guid = hashlib.sha256(clean.encode()).hexdigest()[:16]

            # 1) URL 기반 중복 차단
            if seen_check_and_set_url(guid):
                app.logger.info(f"⏩ already seen(url): {guid}")
                continue

            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            hay = f"{title} {summary}"

            # 2) 하이브리드 매칭 + "≥3 카테고리" 규칙
            matched_labels, dbg = match_categories(title, summary)
            if len(matched_labels) < REQUIRED_MIN_CATS:
                app.logger.info(f"❌ <{REQUIRED_MIN_CATS} categories: {title[:100]}...")
                continue

            # 3) 제목 기반 보조 중복 차단
            if seen_set_title_once(title):
                app.logger.info(f"⏩ already seen(title): {title[:80]}…")
                continue

            app.logger.info(f"✅ MATCH({len(matched_labels)} cats): {title[:120]}... — {matched_labels}")

            article_time = ny_kst_label(
                getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            )
            ticker = extract_ticker(hay)
            company = extract_company(title) or extract_company(summary)
            sector = get_sector_with_cache(ticker, sector_cache)

            try:
                ko = summarize_ko(summary if summary else title)
            except Exception as ex:
                ko = "(요약 실패) " + (summary[:200] or title)
                app.logger.error(f"[gpt] error: {ex}")

            payload = discord_embed(matched_labels, title, url, ticker, company, sector, article_time, ko, dbg)
            push_discord(payload)
            app.logger.info("📤 pushed to Discord")

            # Google Sheets 기록
            row = [
                now_kst, feed_url.split('/')[2], guid,
                ticker or "", company or "", sector, "|".join(matched_labels),
                title, article_time, ko, url
            ]
            try:
                append_sheet(row)
                app.logger.info("[sheets] append ok")
            except Exception as e:
                app.logger.error(f"[sheets] append failed: {e}")

    save_json_dict(EMBED_CACHE_FILE, _embed_cache)
    save_json_dict(SECTOR_CACHE_FILE, sector_cache)
    app.logger.info("🔎 run_once: done")

# -------------------- 백그라운드 루프 --------------------
stop_event = threading.Event()
def scanner_loop():
    app.logger.info("🚀 scanner_loop: started")
    while not stop_event.is_set():
        try:
            run_once()
        except Exception as e:
            app.logger.error(f"[scanner] error: {e}")
        delay = random.uniform(JITTER_MIN, JITTER_MAX)
        app.logger.info(f"⏱️ sleeping {delay:.1f}s")
        time.sleep(delay)

# -------------------- Flask/Healthz --------------------
start_time = time.time()

@app.route("/")
def root():
    return "OK", 200

@app.route("/healthz")
def healthz():
    return jsonify({"status":"ok","uptime_sec": int(time.time()-start_time)}), 200

@app.route("/_sheet_test")
def sheet_test():
    try:
        now = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        append_sheet(["__TEST__", now, "write-from-app", "", "", "", "", "", "", "", ""])
        return "sheet append ok", 200
    except Exception as e:
        app.logger.error(f"[sheets] manual test failed: {e}")
        return f"sheet append failed: {e}", 500

def main():
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT", "8000"))
    app.logger.info(f"🌐 starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()








