# agents/enriched_google_search.py
import json, os, requests
try:
    import tldextract
except ImportError:
    tldextract = None
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID  = os.getenv("GOOGLE_CSE_ID")

def _domain(url: str) -> str:
    try:
        e = tldextract.extract(url)
        return ".".join([p for p in [e.domain, e.suffix] if p])
    except Exception:
        return ""

def EnrichedGoogleSearch(company_name: str, num_results: int = 8) -> dict:
    if not API_KEY or not CSE_ID:
        raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
    q = f"{company_name.strip()} company overview"
    url = "https://www.googleapis.com/customsearch/v1"
    r = requests.get(url, params={
        "key": API_KEY, "cx": CSE_ID, "q": q, "num": min(max(num_results,1),10),
        "hl": "en", "gl": "us", "safe": "active",
    }, timeout=10)
    r.raise_for_status()
    items = (r.json().get("items") or [])
    results = [{
        "title": (it.get("title") or "").strip(),
        "link": it.get("link",""),
        "snippet": (it.get("snippet") or "").strip(),
        "domain": _domain(it.get("link","")),
        "rank": i+1,
    } for i, it in enumerate(items)]

    preferred = {"wikipedia.org","linkedin.com","crunchbase.com","bloomberg.com"}
    def sentence(s):
        s = (s or "").strip()
        s = s.split(".")[0].strip("–—-:;()[] ")
        return s if len(s) > 20 else ""
    one = ""
    for r0 in results:
        if r0["domain"].lower() in preferred:
            one = sentence(r0["snippet"])
            if one: break
    if not one:
        for r0 in results:
            one = sentence(r0["snippet"])
            if one: break
    return {"one_liner": one or "", "results": results}

class SearchArgs(BaseModel):
    company_name: str = Field(..., description="The company name in English if available; Arabic otherwise.")

@tool(args_schema=SearchArgs)
def enriched_google_search(company_name: str) -> str:
    """Fetch enriched Google results for a company; returns JSON with 'one_liner' and 'results'."""
    print(f"[TOOL] enriched_google_search called with: {company_name}")
    try:
        return json.dumps(EnrichedGoogleSearch(company_name), ensure_ascii=False)
    except Exception as e:
        print(f"[TOOL] ERROR: {e}")
        return json.dumps({"one_liner": "", "results": [], "error": str(e)})


def agent_enriched_google_search(llm):
    return create_react_agent(
        model=llm,
        prompt=(
            "You are a company research assistant."
            "When a company name is provided, call `enriched_google_search(company_name=...)`."
            "Read the JSON it returns and immediately send an assistant message:"
            "`**Company insight**: <one_liner>`"
            "Do NOT dump the full results list. Keep it to a single sentence, then continue the flow."
            "Never invent facts; rely only on the tool output."
        ),
        name="company_research_assistant",
        tools=[enriched_google_search],
    )
