"""Answer generation: one typed model invocation under the frozen grounding prompt.

The model arrives as :class:`BaseChatModel`, so CI drives this exact code path with
``GenericFakeChatModel`` and never needs a key. The regenerate-once policy lives in the
service; this module only knows how to assemble messages and run one pass.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from hedis_copilot.answer.prompts import SYSTEM_PROMPT, build_context
from hedis_copilot.retrieval.types import ScoredChunk


def generate(
    model: BaseChatModel,
    query: str,
    chunks: list[ScoredChunk],
    *,
    correction: str | None = None,
) -> str:
    """One generation pass; ``correction`` is the service's regenerate-once hint."""
    human = f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {query}"
    if correction is not None:
        human = f"{human}\n\n{correction}"
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)])
    return str(response.text)
