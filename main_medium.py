import asyncio
import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, END, StateGraph


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


def create_graph(llm):
    class State(TypedDict, total=False):
        query: str
        response: str
        books: list

    async def call_llm(state: State) -> State:
        prompt = (
            f"Answer this question: {state['query']}\n\n"
            "Return a list of book recommendations in NDJSON format. "
            "Each line should be a valid JSON object with: title, author, isbn, genre, and reason.\n"
        )
        content = ""
        async for chunk in llm.astream(prompt):
            piece = getattr(chunk, "content", None)
            if piece:
                if isinstance(piece, str):
                    content += piece
                else:
                    for el in piece:
                        if el.get("type", "") == "text":
                            content += el["text"]
        return {"response": content}

    async def parse_books(state: State) -> State:
        books = []
        for line in state['response'].strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    book = json.loads(line)
                    books.append(book)
                except json.JSONDecodeError:
                    continue
        return {"books": books}

    async def process_books(state: State) -> State:
        processed = []
        for book in state['books']:
            processed_book = process_book(book)
            print(processed_book)
            processed.append(processed_book)
        return {"books": processed}

    workflow = StateGraph(state_schema=State)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("parse_books", parse_books)
    workflow.add_node("process_books", process_books)

    workflow.add_edge(START, "call_llm")
    workflow.add_edge("call_llm", "parse_books")
    workflow.add_edge("parse_books", "process_books")
    workflow.add_edge("process_books", END)

    return workflow.compile()


async def main():
    load_dotenv()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    graph = create_graph(llm)

    query = input("What kind of books are you looking for? ")
    print("\nStreaming recommendations...\n")

    books_count = 0
    async for event in graph.astream({"query": query}, stream_mode="updates"):
        if "process_books" in event:
            books_count = len(event["process_books"].get("books", []))

    print("\n✅ Done!")
    print(f"Found {books_count} books")


if __name__ == "__main__":
    asyncio.run(main())
