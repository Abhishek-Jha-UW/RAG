from __future__ import annotations

import os

from openai import OpenAI


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st

        return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception as exc:  # noqa: BLE001 — streamlit absent in tests/CLI
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set env OPENAI_API_KEY or Streamlit secret OPENAI_API_KEY."
        ) from exc


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())
