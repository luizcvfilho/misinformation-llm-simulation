from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache/vad_lexicon_comparison"
NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_V1_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
MEMOLON_RECORD_URL = "https://zenodo.org/api/records/3756607"
USER_AGENT = "Mozilla/5.0 (compatible; VADLexiconComparison/1.0)"


class RemoteArchive(io.RawIOBase):
    """Read a public ZIP through HTTP ranges without downloading other languages."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.transferred_bytes = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (0, self.position, self.size)[whence]
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid remote archive offset")
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end == self.position:
            return b""
        start = self.position
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end - 1}",
                "Accept-Encoding": "identity",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = f"bytes {start}-{end - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor the exact requested byte range")
            payload = response.read()
        if len(payload) != end - start:
            raise ValueError("Incomplete remote ZIP range")
        self.position = end
        self.transferred_bytes += len(payload)
        return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_nrc_archive(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as source:
            with path.with_suffix(".partial").open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        path.with_suffix(".partial").replace(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"NRC archive CRC failure: {bad_member}")
        print(f"Validated {path.name}: {len(archive.namelist())} members", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache resources for VAD comparisons.")
    parser.add_argument(
        "--include-portuguese-pilot",
        action="store_true",
        help="Also cache NRC v1 and Portuguese MEmoLon for the initial bilingual pilot.",
    )
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nrc_v2_path = PROJECT_ROOT / ".cache/nrc_vad/NRC-VAD-Lexicon-v2.1.zip"
    download_nrc_archive(NRC_URL, nrc_v2_path)
    nrc_path, nrc_url = nrc_v2_path, NRC_URL
    if args.include_portuguese_pilot:
        nrc_path, nrc_url = CACHE_DIR / "NRC-VAD-Lexicon-v1.zip", NRC_V1_URL
        download_nrc_archive(nrc_url, nrc_path)

    with urllib.request.urlopen(MEMOLON_RECORD_URL, timeout=60) as response:
        record = json.load(response)
    (CACHE_DIR / "memolon_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = next(item for item in record["files"] if item["key"] == "MTL_grouped.zip")
    remote = RemoteArchive(metadata["links"]["self"], metadata["size"])
    members = []
    with zipfile.ZipFile(remote) as archive:
        for language in ("pt", "en") if args.include_portuguese_pilot else ("en",):
            info = next(
                item for item in archive.infolist() if Path(item.filename).name == f"{language}.tsv"
            )
            destination = CACHE_DIR / f"memolon_{language}.tsv"
            if not destination.exists():
                print(
                    f"Downloading {info.filename} ({info.compress_size:,} compressed bytes)",
                    flush=True,
                )
                with archive.open(info) as source:
                    with destination.with_suffix(".partial").open("wb") as target:
                        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                destination.with_suffix(".partial").replace(destination)
            members.append(
                dict(
                    language=language,
                    member=info.filename,
                    crc32=f"{info.CRC:08x}",
                    size=info.file_size,
                    sha256=sha256(destination),
                )
            )
            with destination.open(encoding="utf-8") as handle:
                print(language, [next(handle).rstrip() for _ in range(3)], flush=True)
    manifest = dict(
        nrc=dict(url=nrc_url, sha256=sha256(nrc_path), size=nrc_path.stat().st_size),
        memolon=dict(
            record_url=MEMOLON_RECORD_URL,
            archive=metadata,
            members=members,
            bytes_transferred=remote.transferred_bytes,
            verification="ZIP member CRC32 and local SHA256; full archive MD5 not checked.",
        ),
    )
    manifest_name = "download_manifest.json"
    if args.include_portuguese_pilot:
        manifest["nrc_v2"] = dict(
            url=NRC_URL, sha256=sha256(nrc_v2_path), size=nrc_v2_path.stat().st_size
        )
        manifest_name = "pilot_download_manifest.json"
    (CACHE_DIR / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
