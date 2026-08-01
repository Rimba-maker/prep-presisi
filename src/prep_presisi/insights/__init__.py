"""Public API: generate_insight(context: InsightContext) -> str | None."""

import logging
import os

from prep_presisi.entities import InsightContext
from prep_presisi.insights.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def generate_insight(context: InsightContext) -> str | None:
    """Terjemahkan InsightContext jadi 1 kalimat insight via LangChain ChatOpenRouter.
    Graceful fallback (return None) kalau OPENROUTER_API_KEY tidak diset atau API call
    gagal — dashboard tetap jalan normal tanpa kalimat insight. Pengecualian sadar dari
    prinsip fail-loud (TRD §2.5) karena ini dependency eksternal opsional, bukan
    korupsi data internal (TRD §3.2)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_openrouter import ChatOpenRouter

        model_name = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
        model = ChatOpenRouter(model=model_name, api_key=api_key, max_tokens=150)
        response = model.invoke(build_prompt(context))
        return str(response.content).strip()
    except Exception:
        logger.warning("generate_insight() gagal, fallback ke None", exc_info=True)
        return None


__all__ = ["generate_insight"]
