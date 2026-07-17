import json
import logging
from collections.abc import AsyncIterator

import config  # noqa: F401 — load .env for ADK
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.agents.helpers import sources_from_tool_log
from app.agents.tool_loop.loop import run_tool_loop, ToolLoopResult
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.llm.chat.types import ChatProvider
from app.models.auth_model import MavanUser
from app.models.chat_model import MavanChatCompletionRequest, MavanChatCompletionResponse
from app.services.chat_turn import prepare_turn, save_assistant_reply

logger = logging.getLogger(__name__)


async def _stream_turn(request: ChatCompletionRequest, mode: str, sources: list[dict], chat_provider: ChatProvider, chat_id: str, message_id: str, prefixes: tuple[str, ...] | None) -> AsyncIterator[ServerSentEvent]:
    yield ServerSentEvent(event="start", data=json.dumps({"chat_id": chat_id, "message_id": message_id, "mode": mode}))

    try:
        if mode == "agent":
            yield ServerSentEvent(event="thinking", data=json.dumps({"status": "working"}))

        accumulated = ""
        tool_log = []

        if mode == "ask":
            stream_iter = chat_provider.stream(request)
            async for delta in stream_iter:
                accumulated += delta
                yield ServerSentEvent(event="partial", data=accumulated.strip())
        else:
            result = await run_tool_loop(request=request, chat_provider=chat_provider, source_prefixes=prefixes)
            accumulated = result.answer.strip()
            if accumulated:
                yield ServerSentEvent(event="partial", data=accumulated)
            tool_log = result.tool_log
            sources = sources_from_tool_log(result.tool_log)

        await save_assistant_reply(message_id=message_id, content_assistant=accumulated.strip(), mode=mode, sources=sources, tool_log=tool_log, model=request.model)
        yield ServerSentEvent(
            event="completed",
            data=json.dumps({
                "content": accumulated.strip(),
                "mode": mode,
                "sources": sources,
                "tool_log": tool_log,
            }),
        )
    except Exception as e:
        logger.exception("Stream turn failed")
        yield ServerSentEvent(event="error", data=json.dumps({"error": str(e)}))
        raise


async def handle_turn(chat_request: MavanChatCompletionRequest, current_user: MavanUser) -> EventSourceResponse | MavanChatCompletionResponse:
    chat_request, chat_id, message_id, prefixes = await prepare_turn(chat_request=chat_request, current_user=current_user)

    llm_request: ChatCompletionRequest = prepare_chat_request(chat_request, provider_name=chat_request.provider_name)
    chat_provider: ChatProvider = get_capability(Capability.CHAT)(chat_request.provider_name)

    if chat_request.stream:
        return EventSourceResponse(_stream_turn(llm_request, mode=chat_request.mode, sources=chat_request.sources, chat_provider=chat_provider, chat_id=chat_id, message_id=message_id, prefixes=prefixes))

    if chat_request.mode == "ask":
        chat_response: ChatCompletionResponse = await chat_provider.complete(llm_request)
        response = MavanChatCompletionResponse(chat_id=chat_id, message_id=message_id, model=llm_request.model, text=chat_response.text or "", mode="ask", sources=chat_request.sources)
    else:
        result: ToolLoopResult = await run_tool_loop(request=llm_request, chat_provider=chat_provider, source_prefixes=prefixes)
        response = MavanChatCompletionResponse(chat_id=chat_id, message_id=message_id, model=llm_request.model, text=result.answer, mode="agent", sources=sources_from_tool_log(result.tool_log), tool_calls=result.tool_log)

    await save_assistant_reply(message_id=message_id, content_assistant=response.text, mode=response.mode, sources=response.sources, tool_log=response.tool_calls, model=response.model)
    return response
