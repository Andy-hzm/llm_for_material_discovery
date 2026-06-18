import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import Optional

from llm_materials.data.base import BaseSource

ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 100   # max allowed per arXiv request
REQUEST_PAUSE = 3  # seconds between batches (arXiv rate limit)


class ArxivSource(BaseSource):
    """
    Fetch materials-science abstracts from arXiv (cond-mat.mtrl-sci).
    Uses Ray for parallel cleaning; fetching is done in serial batches
    to respect arXiv's rate limit.
    """

    def __init__(self, query: str = "cat:cond-mat.mtrl-sci"):
        self.query = query

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch(self, n: int) -> list[dict]:
        """Paginate the arXiv API to collect up to n abstracts."""
        records = []
        start = 0
        while len(records) < n:
            batch = self._fetch_batch(start, min(BATCH_SIZE, n - len(records)))
            if not batch:
                break
            records.extend(batch)
            start += len(batch)
            print(f"Fetched {len(records)}/{n}")
            if len(batch) == BATCH_SIZE:
                time.sleep(REQUEST_PAUSE)
        return records

    def _fetch_batch(self, start: int, size: int) -> list[dict]:
        params = {
            "search_query": self.query,
            "start": start,
            "max_results": size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        records = []
        for entry in root.findall("atom:entry", ns):
            records.append({
                "id": entry.find("atom:id", ns).text,
                "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                "text": entry.find("atom:summary", ns).text.strip(),
                "published": entry.find("atom:published", ns).text,
                "source": "arxiv",
            })
        return records

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------

    def clean(self, record: dict) -> Optional[dict]:
        text = record["text"]
        # collapse line breaks and excess whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # drop very short abstracts (likely parsing errors)
        if len(text) < 100:
            return None
        return {**record, "text": text}
