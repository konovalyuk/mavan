# import asyncio
# from functools import lru_cache
#
# from llama_models.llama3.api.datatypes import UserMessage, SystemMessage
# from llama_models.llama3.reference_impl.generation import Llama
#
# from config import llm_settings
#
#
# @lru_cache(maxsize=1)
# def _load_generator() -> Llama:
#     return Llama.build(
#         ckpt_dir=llm_settings.LLAMA_CKPT_DIR,
#         max_seq_len=llm_settings.LLAMA_MAX_SEQ_LEN,
#         max_batch_size=llm_settings.LLAMA_MAX_BATCH_SIZE,
#         model_parallel_size=llm_settings.LLAMA_MODEL_PARALLEL_SIZE,
#     )
#
#
# class LlamaLocalProvider:
#     async def stream(self, *, messages, model, max_tokens, temperature):
#         generator = _load_generator()
#
#         dialog = []
#         for m in messages:
#             if m.role == "system":
#                 dialog.append(SystemMessage(role="system", content=m.content))
#             elif m.role == "user":
#                 dialog.append(UserMessage(role="user", content=m.content))
#
#         def _generate():
#             result = generator.chat_completion(
#                 dialog,
#                 max_gen_len=max_tokens or llm_settings.LLAMA_MAX_GEN_LEN,
#                 temperature=temperature if temperature is not None else llm_settings.DEFAULT_TEMPERATURE,
#                 top_p=0.9,
#             )
#             return result.generation.content
#
#         text = await asyncio.to_thread(_generate)
#         # llama_models не стримит нативно — отдаём по словам
#         for word in text.split():
#             yield word + " "
#             await asyncio.sleep(0)