import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime


def process_book(item):
    import random

    if not isinstance(item, dict):
        return item

    required_fields = ['title', 'author', 'isbn', 'genre', 'reason']
    for field in required_fields:
        if field not in item:
            item[field] = ""

    if random.random() < 0.25:
        item["is_in_store"] = "❌"
    else:
        item["is_in_store"] = "✅"

    print(f"📚 {item.get('title')} by {item.get('author')}")
    return item


def get_langchain_prompt():
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate.from_template(
        "Answer this question: {question}\n\n"
        "Return a list of book recommendations in NDJSON format. "
        "Each line should be a valid JSON object with: title, author, isbn, genre, and reason.\n"
    )


def create_graph(llm: BaseLanguageModel) -> Any:
    class RAGState(TypedDict, total=False):
        query: str
        item: dict
        done: bool

    @dataclass
    class Context:
        queue: Optional[asyncio.Queue] = None
        stream_task: Optional[asyncio.Task] = None

    OBJ_LINE = re.compile(
        r"^\s*("
        r"\{"
        r'(?:[^{}\n"]+|"(?:\\.|[^"\\])*")*'
        r"\}"
        r")\s*(?:\n|$)",
        re.DOTALL,
    )
    lc_prompt = get_langchain_prompt()

    async def _stream_reader(
        prompt: str, llm, config, queue: asyncio.Queue
    ):
        buf = ""
        try:
            async for chunk in llm.astream(prompt, config=config):
                piece = getattr(chunk, "content", None)
                if not piece:
                    continue
                if isinstance(piece, str):
                    buf += piece
                else:
                    for el in piece:
                        if el.get("type", "") == "text":
                            buf += el["text"]
                        else:
                            print(f"INVALID TYPE: {piece}")
                            break
                while m := OBJ_LINE.match(buf):
                    raw = m.group(1)
                    buf = buf[m.end() :]
                    obj = json.loads(raw)
                    await queue.put(obj)
        finally:
            await queue.put(None)

    async def start_stream(
        state: RAGState, config, runtime: Runtime[Context]
    ) -> RAGState:
        if runtime.context.queue is None:
            runtime.context.queue = asyncio.Queue()
        if (
            runtime.context.stream_task is None
            or runtime.context.stream_task.done()
        ):
            prompt = lc_prompt.format(question=state["query"])
            runtime.context.stream_task = asyncio.create_task(
                _stream_reader(
                    prompt, llm, config=config, queue=runtime.context.queue
                )
            )
        return {}

    async def next_item(
        state: RAGState, runtime: Runtime[Context]
    ) -> RAGState:
        q = runtime.context.queue
        assert q is not None
        obj = await q.get()
        if obj is None:
            return {"done": True}
        return {"item": obj, "done": False}

    async def consume_item(
        state: RAGState, runtime: Runtime[Context]
    ) -> RAGState:
        if state["done"]:
            return {"done": True}
        item = process_book(state["item"])
        return {"item": item, "done": False}

    def conditional_end(state: RAGState):
        return "end" if state["done"] else "back"

    workflow = StateGraph(state_schema=RAGState, context_schema=Context)
    workflow.add_node("start_stream", start_stream)
    workflow.add_node("next_item", next_item)
    workflow.add_node("consume_item", consume_item)

    workflow.add_edge(START, "start_stream")
    workflow.add_edge("start_stream", "next_item")
    workflow.add_edge("next_item", "consume_item")
    workflow.add_conditional_edges(
        "consume_item",
        conditional_end,
        {"back": "next_item", "end": END},
    )
    graph = workflow.compile()

    return graph


async def main():
    import os
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    graph = create_graph(llm)

    query = input("What kind of books are you looking for? ")
    print("\nStreaming recommendations...\n")

    config = {}
    async for event in graph.astream(
        {"query": query},
        stream_mode="updates",
        config=config,
        context={
            "stream_task": None,
            "queue": None,
        },
    ):
        if datapoint:= event.get('consume_item', False):
            if book_json := datapoint.get("item", False) :
                print(book_json)

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
