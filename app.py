import os, re, time, json, hashlib, requests, feedparser, datetime, threading, random
from zoneinfo import ZoneInfo
from openai import OpenAI
from flask import Flask, jsonify
import yfinance as yf

# ===================== 기본 설정 =====================
RSS_FEEDS = [
    "https://www.prnewswire.com/rss/",                  # 미국 중심
    "https://www.globenewswire.com/rss/list",          # EN 중심(북미 위주)
]

# 키워드 전체 세트 (요청 목록 그대로)
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
share\ repurchase|buyback|repurchase\ program|authorization\ to\ repurchase|
in\ discussions\ to\ acquire|exploring\ acquisition|preliminary\ discussions|
enters\ into\ definitive\ agreement|merger\ agreement|acquisition\ agreement|
land\ use\ permit|\bLUP\b|conditional\ use\ permit|\bCUP\b|zoning\ approval|
federal\ funding|federal\ award|US\ federal|grant\ award|contract\ award)\b''')

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

# Google Sheets는 "옵션": 환경변수가 둘 다 있으면 기록, 아니면 건너뜀
GSHEET_KEY = os.getenv("GSHEET_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# 지터: 35~50초 사이 무작위 대기
JITTER_MIN, JITTER_MAX = 35, 50

# 상태 파일
SEEN_FILE = "seen.json"
SECTOR_CACHE_FILE = "sector_cache.json"

client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== 선택적: Google Sheets =====================
use_sheets = bool(GSHEET_KEY and GOOGLE_SERVICE_ACCOUNT_JSON)
if use_sheets:
    import gspread
    from google.oauth2.service_account import Credentials
    def append_sheet(row):
        creds = Credentials.from_service_account_info(
            json.loads(GOOGLE_SERVICE_ACCOUNT_JSON),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEET_KEY)
        ws = sh.worksheet("news_log")
        ws.append_row(row, value_input_option="USER_ENTERED")
else:
    def append_sheet(row):
        return  # 시트 비활성 모드

# ===================== 유틸 =====================
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

def ny_kst_label(struct_time_or_none):
    # RSS published/updated는 UTC 기준 time.struct_time
    if struct_time_or_none:
        dt_utc = datetime.datetime(*struct_time_or_none[:6], tzinfo=ZoneInfo("UTC"))
    else:
        dt_utc = datetime.datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    dt_ny = dt_utc.astimezone(ZoneInfo("America/New_York"))
    dt_kst = dt_utc.astimezone(ZoneInfo("Asia/Seoul"))
    return f"{dt_ny.strftime('%Y-%m-%d %H:%M:%S ET')} ({dt_kst.strftime('%Y-%m-%d %H:%M:%S KST')})"

def hash_uid(url, title):
    return hashlib.sha256((url + "||" + title).encode("utf-8")).hexdigest()[:16]

# ===================== 티커/회사/섹터 =====================
TICKER_PATTERNS = [
    r"\b(?:NASDAQ|Nasdaq|NYSE|AMEX|OTC|TSX|ASX|HKEX):\s*([A-Z]{1,5})\b",
    r"\$\b([A-Z]{1,5})\b",
    r"\(([A-Z]{1,5})\)",
    r"\[([A-Z]{1,5})\]",
    r"\bTICKER:\s*([A-Z]{1,5})\b",
    r"\b([A-Z]{2,5})\b"  # 마지막 fallback (노이즈 주의)
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
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    cache[ticker] = sector
    return sector

# ===================== GPT 요약/한글 번역 =====================
def summarize_ko(text):
    prompt = (
        "Summarize the press release in 2–3 concise sentences and translate the summary into Korean. "
        "Keep company/product names as-is.\n\nTEXT:\n" + (text or "")[:2000]
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role":"user","content":prompt}],
    )
    return r.choices[0].message.content.strip()

# ===================== Discord 푸시 =====================
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
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[discord] push error: {e}", flush=True)

# ===================== 카테고리 라벨러 =====================
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
]
def classify(text):
    for label, pat in CATEGORIES:
        if re.search(pat, text, re.I): return label
    return "other"

# ===================== 스캔 1회 =====================
def run_once():
    seen = load_json_set(SEEN_FILE)
    sector_cache = load_json_dict(SECTOR_CACHE_FILE)
    now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for e in feed.entries[:80]:
            url = getattr(e, "link", "")
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            guid = getattr(e, "id", "") or hash_uid(url, title)
            if guid in seen:
                continue

            hay = f"{title} {summary}"
            if not KEYWORDS.search(hay):
                continue

            article_time = ny_kst_label(getattr(e,"published_parsed", None) or getattr(e,"updated_parsed", None))
            cat = classify(hay)
            ticker = extract_ticker(hay)
            company = extract_company(title) or extract_company(summary)
            sector = get_sector_with_cache(ticker, sector_cache)

            try:
                ko = summarize_ko(summary if summary else title)
            except Exception as ex:
                ko = "(요약 실패) " + (summary[:200] or title)
                print(f"[gpt] error: {ex}", flush=True)

            payload = discord_embed(cat, title, url, ticker, company, sector, article_time, ko)
            push_discord(payload)

            # 시트 기록 (옵션)
            row = [
                now_kst, feed_url.split('/')[2], guid,
                ticker or "", company or "", sector, cat,
                title, article_time, ko, url
            ]
            append_sheet(row)

            seen.add(guid)

    save_json_set(SEEN_FILE, seen)
    save_json_dict(SECTOR_CACHE_FILE, sector_cache)

# ===================== 백그라운드 루프(지터) =====================
stop_event = threading.Event()
def scanner_loop():
    while not stop_event.is_set():
        try:
            run_once()
        except Exception as e:
            print(f"[scanner] error: {e}", flush=True)
        delay = random.uniform(JITTER_MIN, JITTER_MAX)  # 35~50초
        time.sleep(delay)

# ===================== Flask 헬스체크 =====================
import threading
import os
import time
from flask import Flask, jsonify

app = Flask(__name__)
start_time = time.time()

@app.route("/")
def root():
    return "OK", 200

@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "uptime_sec": int(time.time() - start_time)
    }), 200

def main():
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

