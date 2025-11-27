import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

print("Generating graph visualizations...\n")

from main_medium import create_graph as create_medium_graph
medium_graph = create_medium_graph(llm)

mermaid_medium = medium_graph.get_graph().draw_mermaid()
with open("graph_medium.mmd", "w") as f:
    f.write(mermaid_medium)
print("✅ Saved graph_medium.mmd")

try:
    png_medium = medium_graph.get_graph().draw_mermaid_png()
    with open("graph_medium.png", "wb") as f:
        f.write(png_medium)
    print("✅ Saved graph_medium.png")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")

from main import create_graph as create_main_graph
main_graph = create_main_graph(llm)

mermaid_main = main_graph.get_graph().draw_mermaid()
with open("graph_main.mmd", "w") as f:
    f.write(mermaid_main)
print("✅ Saved graph_main.mmd")

try:
    png_main = main_graph.get_graph().draw_mermaid_png()
    with open("graph_main.png", "wb") as f:
        f.write(png_main)
    print("✅ Saved graph_main.png")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")

print("\n✅ Done! Open .mmd files in a Mermaid viewer or .png files directly.")
