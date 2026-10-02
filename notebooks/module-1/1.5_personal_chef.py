from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for current recipes, ingredients, and cooking techniques."""
    return tavily_client.search(query, max_results=4)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain.agents import create_agent

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
    max_output_tokens=700,
    timeout=30,
    max_retries=2,
)

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)
