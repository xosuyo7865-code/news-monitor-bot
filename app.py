# app.py — Newswire Scanner (PRNewswire + GlobeNewswire)
# - 35–50s jitter loop
# - Keyword filtering
# - GPT 한국어 요약 (영문 제거)
# - Discord Webhook 푸시
# - (옵션) Google Sheets 로그
# - Sector 캐시 (yfinance)
# - /healthz 헬스체크
# - 중복 방지: Redis(권장) + 로컬 폴백
# - 상세 로그

import os, re, time, json, hashlib, requests, feedparser, datetime, threading, random, logging, socket
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from openai import OpenAI
from flask import Flask, jsonify
import yfinance as yf

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
    "https://www.prnewswire.com/rss/news-releases-list.rss",  # PR Newswire 전체
    "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States"
]

# -------------------- 키워드 --------------------
KEYWORDS = re.compile(r'''(?ix)\b(
pilot\ project|proof[-\ ]of[-\ ]concept|\bPOC\b|pilot|proof-of-concept|kickoff|
supply\ (agreement|contract|deal|order)|
sales\ (agreement|contract)|purchase\ order|\bPO\b|
new\ customer|selects|chooses|deploys|adopts|rollout\ with|goes\ live\ with|
strategic\ partnership|partnership|collaboration|alliances|joint\ venture|\bJV\b|
contract\ with|awarded\ by|awards\ contract|RFP\ award|\bDOI\b|US\ Army|USAF|Navy|municipality|county|state\ of|
private\ placement|at[-\ ]the[-\ ]market\ private\ placement|equity\ financing|
\bPIPE\b|private\ investment\ in\ public\ equity|
registered\ direct\ offering|\bRDO\b|
phase\ 1\b|phase\ 2\b|phase\ 3\b|clinical\ trial|trial\ results|FDA\ approval|BLA\ approval|NDA\ approval|clearance|
grant|government\ grant|\bDOE\b|DoD\ grant|\bNSF\b|\bSBIR\b|\bARPA-E\b|\bNTIA\b|CHIPS\ Act|
launches|introduces|announces\ new|releases|unveil\w*|debuts|rolls\ out|solution|
\bELOC\b|equity\ line\ of\ credit|standby\ equity\ purchase|
strategic\ investment|equity\ investment|takes\ stake|
purchase\ agreement|asset\ purchase|stock\ purchase|
license\ agreement|licensing\ agreement|licensing\ (deal|contract|partnership)|exclusive\ license|non[-\ ]exclusive\ license|license\ rights|licensing\ rights|
share\ repurchase|buyback|repurchase\ program|authorization\ to\ repurchase|
in\ discussions\ to\ acquire|exploring\ acquisition|preliminary\ discussions|
enters\ into\ definitive\ agreement|merger\ agreement|acquisition\ agreement|
land\ use\ permit|\bLUP\b|conditional\ use\ permit|\bCUP\b|zoning\ approval|
federal\ funding|federal\ award|US\ federal|grant\ award|contract\ award)\b''')

# 카테고리 라벨러(알림 제목에 사용)
CATEGORIES = [
  ("new_project", r"pilot project|proof[- ]of[- ]concept|\bPOC\b|pilot|proof-of-concept|kickoff"),
  ("supply_agreement", r"supply (agreement|contract|deal|order)"),
  ("sales_contract", r"sales (agreement|contract)|purchase order|\bPO\b"),
  ("new_customer", r"new customer|selects|chooses|deploys|adopts|rollout with|goes live with"),
  ("strategic_partnership", r"strategic partnership|partnership|collaboration|alliances|joint venture|\bJV\b"),
  ("gov_public_contract", r"contract with|awarded by|awards contract|RFP award|\bDOI\b|US Army|USAF|Navy|municipality|county|state of"),
  ("private_placement", r"private placement|at[- ]the[- ]market private placement|equity financing"),
  ("pipe", r"\bPIPE\b|private investment in public equity"),
  ("registered_direct", r"registered direct offering|\bRDO\b"),
  ("clinical_result_or_approval", r"phase 1\b|phase 2\b|phase 3\b|clinical trial|trial results|FDA approval|BLA approval|NDA approval|clearance"),
  ("gov_grant_or_policy_fund", r"grant|government grant|\bDOE\b|DoD grant|\bNSF\b|\bSBIR\b|\bARPA-E\b|\bNTIA\b|CHIPS Act"),
  ("new_product_or_tech_announcement", r"launches|introduces|announces new|releases|unveil\w*|debuts|rolls out|solution"),
  ("eloc", r"\bELOC\b|equity line of credit|standby equity purchase"),
  ("equity_or_strategic_investment", r"strategic investment|equity investment|takes stake"),
  ("purchase_agreement", r"purchase agreement|asset purchase|stock purchase"),
  ("buyback", r"share repurchase|buyback|repurchase program|authorization to repurchase"),
  ("mna_discussion", r"in discussions to acquire|exploring acquisition|preliminary discussions"),
  ("mna_negotiation", r"enters into definitive agreement|merger agreement|acquisition agreement"),
  ("land_use_permit", r"land use permit|\bLUP\b|conditional use permit|\bCUP\b|zoning approval"),
  ("federal_funding", r"federal funding|federal award|US federal|grant award|contract award"),
  ("license_agreement", r"license agreement|licensing agreement|licensing (deal|contract|partnership)|exclusive license|non[- ]exclusive license|license rights|licensing rights"),
]
def classify(text):
    for label, pat in CATEGORIES:
        if re.search(pat, text, re.I): return label
    return "other"

# -------------------- 파일 경로/지터 --------------------
SEEN_FILE = "seen.json"
SECTOR_CACHE_FILE = "sector_cache.json"
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

# -------------------- GPT 요약(→한국어) --------------------
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

# -------------------- Discord --------------------
def discord_embed(category, title, url, ticker, company, sector, article_time, summary_ko):
    return {
        "username": "Newswire Scanner",
        "embeds": [{
            "title": f"[{category}] {title}",
            "url": url,
            "description": f"{summary_ko}\n\n🔗 원문: {url}",
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
    import redis  # render의 이미지에 없으면 ImportError → 자동 폴백
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
    t = re.sub(r"[\W_]+", " ", t)     # 특수문자 제거
    t = re.sub(r"\s+", " ", t).strip()
    return t

def seen_check_and_set_url(guid: str, ttl_seconds: int = 14*24*3600) -> bool:
    """
    URL GUID 기준 체크. True → 이미 봄 / False → 처음 봄(마킹 성공)
    """
    r = _get_redis()
    if r:
        key = f"nwscanner:seen:url:{guid}"
        try:
            ok = r.setnx(key, INSTANCE)  # race-free
            if ok: r.expire(key, ttl_seconds)
            return not ok
        except Exception as e:
            app.logger.error(f"[dedup] Redis error(url): {e}")
    # 폴백: 로컬 파일 세트
    seen = load_json_set(SEEN_FILE)
    if guid in seen: return True
    seen.add(guid); save_json_set(SEEN_FILE, seen)
    return False

def seen_set_title_once(title: str, ttl_seconds: int = 14*24*3600) -> bool:
    """
    제목 기반 보조 차단. 이미 있으면 True, 없으면 set 후 False.
    (URL이 달라도 같은 제목이면 차단)
    """
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
    # 로컬 폴백은 URL 세트와 충돌 우려로 생략 (Redis 권장)
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

            # ✅ 1차: URL 기준 중앙 중복 차단 (즉시)
            if seen_check_and_set_url(guid):
                app.logger.info(f"⏩ already seen(url): {guid}")
                continue

            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            hay = f"{title} {summary}"

            if not KEYWORDS.search(hay):
                app.logger.info(f"❌ no match: {title[:100]}...")
                continue

            # ✅ 2차: 제목 기반 보조 차단 (서로 다른 URL이라도 같은 기사면 차단)
            if seen_set_title_once(title):
                app.logger.info(f"⏩ already seen(title): {title[:80]}…")
                continue

            app.logger.info(f"✅ MATCH: {title[:120]}...")

            article_time = ny_kst_label(
                getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            )
            cat = classify(hay)
            ticker = extract_ticker(hay)
            company = extract_company(title) or extract_company(summary)
            sector = get_sector_with_cache(ticker, sector_cache)

            try:
                ko = summarize_ko(summary if summary else title)
            except Exception as ex:
                ko = "(요약 실패) " + (summary[:200] or title)
                app.logger.error(f"[gpt] error: {ex}")

            payload = discord_embed(cat, title, url, ticker, company, sector, article_time, ko)
            push_discord(payload)
            app.logger.info("📤 pushed to Discord")

            # Google Sheets 기록
            row = [
                now_kst, feed_url.split('/')[2], guid,
                ticker or "", company or "", sector, cat,
                title, article_time, ko, url
            ]
            try:
                append_sheet(row)
                app.logger.info("[sheets] append ok")
            except Exception as e:
                app.logger.error(f"[sheets] append failed: {e}")

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

# 진단용 시트 테스트
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








