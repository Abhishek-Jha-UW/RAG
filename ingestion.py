from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import docx
import pandas as pd
from PyPDF2 import PdfReader

Chunk = dict[str, Any]


def _sha24(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()[:24]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _profile_dataframe(df: pd.DataFrame, source: str, sheet: str | None) -> str:
    title = f"Dataset profile: {source}"
    if sheet:
        title += f" | Sheet: {sheet}"
    lines = [title, f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "", "Columns (dtype):"]
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    nulls = df.isnull().sum()
    if nulls.any():
        lines += ["", "Null counts (non-zero):"]
        for col, n in nulls[nulls > 0].items():
            lines.append(f"  - {col}: {int(n)}")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        desc = df[num_cols].describe().T
        lines += ["", "Numeric summary (describe):"]
        lines.append(desc.to_string())
    return "\n".join(lines)


def _tabular_row_chunks(
    df: pd.DataFrame,
    source: str,
    sheet: str | None,
    rows_per_chunk: int,
) -> list[Chunk]:
    if df.empty:
        return []
    chunks: list[Chunk] = []
    n = len(df)
    header = " | ".join(str(c) for c in df.columns)
    start = 0
    while start < n:
        end = min(start + rows_per_chunk, n)
        sub = df.iloc[start:end]
        body = sub.to_csv(index=False)
        sheet_label = sheet or "(csv)"
        text = (
            f"Source file: {source}\n"
            f"Sheet / segment: {sheet_label}\n"
            f"Columns: {header}\n"
            f"Row range (1-based): {start + 1}-{end} of {n}\n\n"
            f"{body}"
        )
        chunks.append(
            {
                "text": text,
                "source": source,
                "sheet": sheet_label,
                "page": None,
                "row_range": f"{start + 1}-{end}",
                "kind": "tabular",
            }
        )
        start = end
    return chunks


def _read_pdf_bytes(data: bytes, source: str) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        parts: list[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            parts.append(f"\n[Page {i + 1}]\n{page.get_text() or ''}")
        doc.close()
        return "\n".join(parts).strip()
    except Exception:
        reader = PdfReader(io.BytesIO(data))
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n[Page {i + 1}]\n{page_text}"
        return text.strip()


def _read_docx_bytes(data: bytes, source: str) -> str:
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_plain_text_bytes(data: bytes, source: str) -> str:
    return data.decode("utf-8", errors="replace")


def ingest_bytes(
    name: str,
    data: bytes,
    *,
    tabular_rows_per_chunk: int,
    chunk_words: int = 300,
    chunk_overlap: int = 50,
) -> tuple[list[Chunk], str | None]:
    """Return (chunks, error_message)."""
    name_l = name.lower()
    err: str | None = None
    out: list[Chunk] = []

    try:
        if name_l.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
            out.append(
                {
                    "text": _profile_dataframe(df, name, None),
                    "source": name,
                    "sheet": None,
                    "page": None,
                    "row_range": None,
                    "kind": "profile",
                }
            )
            out.extend(_tabular_row_chunks(df, name, None, tabular_rows_per_chunk))

        elif name_l.endswith((".xlsx", ".xlsm")):
            xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                out.append(
                    {
                        "text": _profile_dataframe(df, name, sheet),
                        "source": name,
                        "sheet": sheet,
                        "page": None,
                        "row_range": None,
                        "kind": "profile",
                    }
                )
                out.extend(_tabular_row_chunks(df, name, sheet, tabular_rows_per_chunk))

        elif name_l.endswith(".pdf"):
            text = _read_pdf_bytes(data, name)
            if not text.strip():
                return [], "No extractable text (try a text-based PDF; scanned PDFs need OCR)."
            out.extend(
                chunk_narrative(
                    text,
                    name,
                    sheet=None,
                    page=None,
                    kind="narrative",
                    chunk_words=chunk_words,
                    overlap=chunk_overlap,
                )
            )

        elif name_l.endswith(".docx"):
            text = _read_docx_bytes(data, name)
            if not text.strip():
                return [], "Empty or unreadable Word document."
            out.extend(
                chunk_narrative(
                    text,
                    name,
                    sheet=None,
                    page=None,
                    kind="narrative",
                    chunk_words=chunk_words,
                    overlap=chunk_overlap,
                )
            )

        elif name_l.endswith(".txt"):
            text = _read_plain_text_bytes(data, name)
            out.extend(
                chunk_narrative(
                    text,
                    name,
                    sheet=None,
                    page=None,
                    kind="narrative",
                    chunk_words=chunk_words,
                    overlap=chunk_overlap,
                )
            )

        else:
            return [], f"Unsupported extension for {name}. Use csv, xlsx, pdf, docx, or txt."

    except Exception as e:  # noqa: BLE001
        return [], f"Error reading {name}: {e}"

    out = dedupe_chunks(out)
    for i, c in enumerate(out):
        c["chunk_id"] = i
    return out, err


def ingest_uploaded_file(
    file,
    tabular_rows_per_chunk: int,
    *,
    chunk_words: int = 300,
    chunk_overlap: int = 50,
) -> tuple[list[Chunk], str | None]:
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass
    return ingest_bytes(
        file.name,
        raw,
        tabular_rows_per_chunk=tabular_rows_per_chunk,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )


def chunk_narrative(
    text: str,
    source: str,
    *,
    sheet: str | None,
    page: str | None,
    kind: str,
    chunk_words: int,
    overlap: int,
) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    segments: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush_buf() -> None:
        nonlocal buf, buf_words
        if buf:
            segments.append("\n\n".join(buf))
            buf = []
            buf_words = 0

    for para in paragraphs:
        wcount = len(_tokenize_words(para))
        if wcount > chunk_words:
            flush_buf()
            segments.extend(_sliding_word_chunks(para, chunk_words, overlap))
            continue
        if buf_words + wcount > chunk_words and buf:
            flush_buf()
        buf.append(para)
        buf_words += wcount
        if buf_words >= chunk_words:
            flush_buf()
    flush_buf()

    chunks: list[Chunk] = []
    for seg in segments:
        if len(_tokenize_words(seg)) > chunk_words * 1.2:
            for sub in _sliding_word_chunks(seg, chunk_words, overlap):
                chunks.append(_chunk_dict(sub, source, sheet, page, kind))
        else:
            chunks.append(_chunk_dict(seg, source, sheet, page, kind))
    return chunks


def _chunk_dict(text: str, source: str, sheet: str | None, page: str | None, kind: str) -> Chunk:
    return {
        "text": text.strip(),
        "source": source,
        "sheet": sheet,
        "page": page,
        "row_range": None,
        "kind": kind,
    }


def _sliding_word_chunks(text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    out: list[str] = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + chunk_words])
        if piece.strip():
            out.append(piece)
        if i + chunk_words >= len(words):
            break
    return out


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        h = _sha24(c.get("text", ""))
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out


def assign_chunk_ids(chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks):
        c["chunk_id"] = i


def corpus_manifest(chunks: list[Chunk]) -> list[dict[str, Any]]:
    by_file: dict[str, dict[str, Any]] = {}
    for c in chunks:
        src = c.get("source", "?")
        if src not in by_file:
            by_file[src] = {"source": src, "chunks": 0, "chars": 0, "kinds": set()}
        by_file[src]["chunks"] += 1
        by_file[src]["chars"] += len(c.get("text", ""))
        by_file[src]["kinds"].add(c.get("kind", "unknown"))
    rows = []
    for v in by_file.values():
        kinds = v.pop("kinds")
        v["kinds"] = ", ".join(sorted(kinds))
        rows.append(v)
    return sorted(rows, key=lambda x: x["source"].lower())


def load_demo_chunks(
    sample_dir: str | Path,
    tabular_rows_per_chunk: int,
    *,
    chunk_words: int = 300,
    chunk_overlap: int = 50,
) -> tuple[list[Chunk], list[str]]:
    """Load packaged sample files from disk (Streamlit-safe paths)."""
    d = Path(sample_dir)
    errors: list[str] = []
    all_chunks: list[Chunk] = []
    for path in sorted(d.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".txt", ".xlsx", ".pdf", ".docx"}:
            continue
        data = path.read_bytes()
        chunks, err = ingest_bytes(
            path.name,
            data,
            tabular_rows_per_chunk=tabular_rows_per_chunk,
            chunk_words=chunk_words,
            chunk_overlap=chunk_overlap,
        )
        if err:
            errors.append(err)
        all_chunks.extend(chunks)
    all_chunks = dedupe_chunks(all_chunks)
    assign_chunk_ids(all_chunks)
    return all_chunks, errors
