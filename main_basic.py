import asyncio
import json
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


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


async def get_book_recommendations(query: str, llm):
    prompt = (
        f"Answer this question: {query}\n\n"
        "Return a list of book recommendations in NDJSON format. "
        "Each line should be a valid JSON object with: title, author, isbn, genre, and reason.\n"
    )

    response = await llm.ainvoke(prompt)
    content = response.content

    books = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                book = json.loads(line)
                processed_book = process_book(book)
                books.append(processed_book)
                print(processed_book)
            except json.JSONDecodeError:
                continue

    return books


async def main():
    load_dotenv()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    query = input("What kind of books are you looking for? ")
    print("\nGetting recommendations...\n")

    books = await get_book_recommendations(query, llm)

    print("\n✅ Done!")
    print(f"Found {len(books)} books")


if __name__ == "__main__":
    asyncio.run(main())
