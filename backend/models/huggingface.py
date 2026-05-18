"""HuggingFace search + per-repo file listing.

Thin wrapper over the public HF API:
* `GET https://huggingface.co/api/models?search=...&filter=gguf`  — repo search
* `GET https://huggingface.co/api/models/{repo}/tree/main?recursive=true` — file list with sizes

We don't authenticate; only public repos are searchable. That's fine — every
mainstream quant publisher (bartowski, mradermacher, TheBloke, unsloth) ships
public repos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx


HF_API = "https://huggingface.co/api"


@dataclass
class HFRepo:
    id: str                    # "owner/name"
    likes: int
    downloads: int
    last_modified: Optional[str]
    pipeline_tag: Optional[str]
    tags: List[str]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "likes": self.likes,
            "downloads": self.downloads,
            "lastModified": self.last_modified,
            "pipelineTag": self.pipeline_tag,
            "tags": self.tags,
        }


@dataclass
class HFFile:
    name: str                  # filename within the repo (may contain /)
    size_bytes: int
    download_url: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "sizeBytes": self.size_bytes,
            "sizeGb": self.size_bytes / (1024 ** 3),
            "downloadUrl": self.download_url,
        }


async def search_repos(query: str, limit: int = 25) -> List[HFRepo]:
    params = {
        "search": query,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{HF_API}/models", params=params)
        r.raise_for_status()
        data = r.json()

    out: List[HFRepo] = []
    for item in data:
        out.append(HFRepo(
            id=item.get("id") or item.get("modelId") or "",
            likes=int(item.get("likes") or 0),
            downloads=int(item.get("downloads") or 0),
            last_modified=item.get("lastModified"),
            pipeline_tag=item.get("pipeline_tag"),
            tags=list(item.get("tags") or []),
        ))
    return out


async def list_gguf_files(repo: str) -> List[HFFile]:
    """Return all `.gguf` files in a repo at HEAD of `main`, with sizes."""
    if "/" not in repo:
        raise ValueError("repo must be 'owner/name'")
    async with httpx.AsyncClient(timeout=15.0) as client:
        # tree endpoint with recursive=true returns sizes via LFS metadata
        r = await client.get(
            f"{HF_API}/models/{repo}/tree/main",
            params={"recursive": "true", "expand": "true"},
        )
        r.raise_for_status()
        items = r.json()

    out: List[HFFile] = []
    for it in items:
        if it.get("type") != "file":
            continue
        path: str = it.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        # Size can be top-level "size" OR under "lfs.size" for LFS-backed files
        size = it.get("size")
        if size is None and isinstance(it.get("lfs"), dict):
            size = it["lfs"].get("size")
        out.append(HFFile(
            name=path,
            size_bytes=int(size or 0),
            download_url=f"https://huggingface.co/{repo}/resolve/main/{path}",
        ))
    # Sort by size (smallest first → cheapest quant on top usually)
    out.sort(key=lambda f: f.size_bytes)
    return out
