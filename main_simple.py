import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


async def main():
    load_dotenv()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    query = input("What kind of books are you looking for? ")
    print("\nStreaming response...\n")

    prompt = (
        f"Answer this question: {query}\n\n"
        "Return a list of book recommendations in NDJSON format. "
        "Each line should be a valid JSON object with: title, author, isbn, genre, and reason.\n"
    )

    async for chunk in llm.astream(prompt):
        content = getattr(chunk, "content", None)
        if content:
            if isinstance(content, str):
                print(content, end="", flush=True)
            else:
                for el in content:
                    if el.get("type", "") == "text":
                        print(el["text"], end="", flush=True)

    print("\n\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
