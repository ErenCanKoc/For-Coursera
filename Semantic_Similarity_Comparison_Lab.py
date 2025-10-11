import os, re, time, json, asyncio, random, csv, xml.etree.ElementTree as ET, sqlite3, hashlib
from datetime import datetime
from urllib.parse import urlparse
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
from tqdm.asyncio import tqdm_asyncio

# ===================== CONFIG =====================
SITEMAP_PATH = "sitemap_canva.xml"
SAMPLE_SIZE = 20
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 384
EMBED_CALL_INTERVAL = 1.2
MAX_CONTAINER_CHARS = 12000
GROUP_ID = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LANG_PATTERN = r"^[a-z]{2}(_[a-z]{2})?$"
SQLITE_CACHE = "embedding_cache.sqlite"
PARALLEL_FETCH = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_last_call = 0.0

# ===================== DB CACHE =====================
def init_cache():
    con = sqlite3.connect(SQLITE_CACHE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            text_hash TEXT PRIMARY KEY,
            embedding TEXT
        )
    """)
    con.commit()
    return con

def get_cached_embedding(con, text_hash):
    cur = con.cursor()
    cur.execute("SELECT embedding FROM cache WHERE text_hash=?", (text_hash,))
    row = cur.fetchone()
    if row:
        return np.array(json.loads(row[0]), dtype=np.float32)
    return None

def save_cached_embedding(con, text_hash, emb):
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO cache(text_hash, embedding) VALUES (?,?)",
                (text_hash, json.dumps(emb.tolist())))
    con.commit()

def hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ===================== HELPERS =====================
def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9À-ž\s]", "", text)
    return text.strip().lower()

def embed_text_cached(con, text):
    global _last_call
    text = normalize_text(text)
    h = hash_text(text)
    cached = get_cached_embedding(con, h)
    if cached is not None:
        return cached
    dt = time.time() - _last_call
    if dt < EMBED_CALL_INTERVAL:
        time.sleep(EMBED_CALL_INTERVAL - dt)
    res = client.embeddings.create(model=EMBED_MODEL, input=text, dimensions=EMBED_DIM)
    _last_call = time.time()
    emb = np.array(res.data[0].embedding, dtype=np.float32)
    save_cached_embedding(con, h, emb)
    return emb

def tfidf_cosine(a, b):
    try:
        X = TfidfVectorizer(min_df=1, max_df=0.95).fit_transform([a,b])
        return float((X[0] @ X[1].T).toarray()[0,0])
    except:
        return 0.0

def greedy_chunk_score(A, B):
    if A.size==0 or B.size==0: return 0.0
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True)+1e-9)
    B /= (np.linalg.norm(B, axis=1, keepdims=True)+1e-9)
    S = A @ B.T
    picks=[]
    while True:
        i,j=np.unravel_index(np.argmax(S),S.shape)
        if S[i,j]<0.2: break
        picks.append(S[i,j])
        S[i,:]=-1; S[:,j]=-1
        if len(picks)>=min(len(A),len(B)): break
    return float(np.median(picks)) if picks else 0.0

def get_root_lang(code):
    return code.split("_")[0] if "_" in code else code

# ===================== FETCH =====================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8,pt;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"
}

async def fetch_html(session, url):
    for variant in [url, url + "?format=json", url + "?_escaped_fragment_="]:
        try:
            async with session.get(variant, timeout=30) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    if "<html" in text or len(text) > 500:
                        return text
        except:
            continue
    return None

# ===================== SITEMAP =====================
def parse_sitemap(path):
    tree = ET.parse(path)
    return [loc.text.strip() for loc in tree.findall(".//{*}loc")]

def extract_lang_and_slug(url):
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts)>1 and re.match(LANG_PATTERN, parts[0], re.IGNORECASE):
        lang = parts[0].lower()
        slug = "/".join(parts[1:])
    else:
        lang = "default"
        slug = "/".join(parts)
    return lang, slug

def group_similar_slugs(urls, threshold=0.85, con=None):
    print("🧩 Slug embedding hesaplanıyor...")
    slug_data = []
    for u in tqdm_asyncio(urls, desc="Slug embedding ilerlemesi"):
        lang, slug = extract_lang_and_slug(u)
        emb = embed_text_cached(con, slug.replace("-", " ").replace("/", " "))
        slug_data.append({"url": u, "lang": lang, "slug": slug, "emb": emb})
    all_embs = np.vstack([s["emb"] for s in slug_data])
    sim = cosine_similarity(all_embs)
    visited=set()
    groups=[]
    for i,s in enumerate(slug_data):
        if i in visited: continue
        group=[s]
        visited.add(i)
        for j in range(i+1,len(slug_data)):
            if j not in visited and sim[i,j]>=threshold:
                group.append(slug_data[j])
                visited.add(j)
        groups.append(group)
    print(f"✅ {len(groups)} içerik grubu bulundu.")
    return groups

# ===================== STRUCTURAL ANALYSIS =====================
def extract_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    removed_tags = soup(["script","style","noscript","form","aside"])
    removed_text_len = sum(len(t.get_text()) for t in removed_tags)
    for t in removed_tags: t.extract()
    total_len = len(soup.get_text())
    removed_ratio = removed_text_len / (total_len + 1e-9)
    header = soup.find("header")
    main = soup.find("main") or soup.body
    footer = soup.find("footer")
    def clean_section(tag):
        return normalize_text(tag.get_text(" "))[:MAX_CONTAINER_CHARS] if tag else ""
    return {
        "header": clean_section(header),
        "main": clean_section(main),
        "footer": clean_section(footer),
        "removed_ratio": round(removed_ratio,3)
    }

async def process_page(session, url):
    html = await fetch_html(session, url)
    if not html: 
        return None
    return extract_sections(html)

def similarity_scores(con, secA, secB):
    scores = {}
    for part in ["header","main","footer"]:
        embA = embed_text_cached(con, secA[part]) if secA[part] else np.zeros((EMBED_DIM,))
        embB = embed_text_cached(con, secB[part]) if secB[part] else np.zeros((EMBED_DIM,))
        emb_score = float(np.dot(embA, embB) / ((np.linalg.norm(embA)*np.linalg.norm(embB))+1e-9))
        tfidf_score = tfidf_cosine(secA[part], secB[part])
        scores[part] = round(0.8*emb_score + 0.2*tfidf_score,3)
    return scores

# ===================== MAIN =====================
async def run():
    con_cache = init_cache()
    urls = parse_sitemap(SITEMAP_PATH)
    groups = group_similar_slugs(urls, threshold=0.85, con=con_cache)
    sampled = random.sample(groups, min(SAMPLE_SIZE, len(groups)))
    print(f"🌍 {len(sampled)} slug grubu analiz edilecek.")
    
    results = []
    connector = aiohttp.TCPConnector(limit_per_host=PARALLEL_FETCH)
    async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
        for g in tqdm_asyncio(sampled, desc="Slug grup analizi ilerlemesi"):
            slug = g[0]['slug']
            pages = []
            tasks = [process_page(session, s['url']) for s in g]
            fetched = await asyncio.gather(*tasks)
            for s, data in zip(g, fetched):
                if data:
                    pages.append({"lang": s['lang'], "url": s['url'], **data})
            for i in range(len(pages)):
                for j in range(i+1, len(pages)):
                    A, B = pages[i], pages[j]
                    scores = similarity_scores(con_cache, A, B)
                    results.append({
                        "slug": slug,
                        "lang1": A["lang"],
                        "lang2": B["lang"],
                        "header_sim": scores["header"],
                        "main_sim": scores["main"],
                        "footer_sim": scores["footer"],
                        "removed_ratio": round(abs(A["removed_ratio"] - B["removed_ratio"]),3)
                    })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"structural_localization_report_cache_resume_{ts}.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slug","lang1","lang2","header_sim","main_sim","footer_sim","removed_ratio"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Smart Cache + Progress raporu kaydedildi: {csv_path}")

if __name__ == "__main__":
    asyncio.run(run())