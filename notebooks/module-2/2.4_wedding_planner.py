"""A LangGraph Studio-ready multi-agent destination wedding planner."""

from pathlib import Path
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
    max_output_tokens=900,
    timeout=30,
    max_retries=2,
)
tavily_client = TavilyClient()
catalog_path = Path(__file__).parent / "resources" / "Chinook.db"


@tool
def search_current_information(query: str) -> str:
    """Search current web information for venues, flights, travel requirements, and local services."""
    results = tavily_client.search(query, max_results=4)
    return str(results.get("results", results))


@tool
def find_music_by_genre(genre: str) -> str:
    """Find up to ten songs from the local music catalog for a requested wedding genre."""
    with sqlite3.connect(catalog_path) as connection:
        rows = connection.execute(
            """
            SELECT tracks.Name, artists.Name, genres.Name
            FROM tracks
            JOIN albums ON tracks.AlbumId = albums.AlbumId
            JOIN artists ON albums.ArtistId = artists.ArtistId
            JOIN genres ON tracks.GenreId = genres.GenreId
            WHERE lower(genres.Name) LIKE lower(?)
            LIMIT 10
            """,
            (f"%{genre}%",),
        ).fetchall()
    return "\n".join(f"{track} - {artist} ({music_genre})" for track, artist, music_genre in rows) or "No matching tracks found."


travel_agent = create_agent(
    model=model,
    tools=[search_current_information],
    system_prompt="You are a travel specialist. Find practical current flight and arrival information, state dates and limitations, and return a compact shortlist.",
)
venue_agent = create_agent(
    model=model,
    tools=[search_current_information],
    system_prompt="You are a wedding venue specialist. Search for venues matching the stated destination, guest count, budget, and style. Cite useful source links from tool results.",
)
music_agent = create_agent(
    model=model,
    tools=[find_music_by_genre],
    system_prompt="You are a wedding music curator. Use the catalog tool and produce a cohesive playlist recommendation for the requested mood.",
)


def run_specialist(agent, request: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=request)]})
    return str(result["messages"][-1].content)


@tool
def get_travel_options(request: str) -> str:
    """Delegate current travel and flight research to the travel specialist."""
    return run_specialist(travel_agent, request)


@tool
def get_venue_options(request: str) -> str:
    """Delegate venue research to the wedding venue specialist."""
    return run_specialist(venue_agent, request)


@tool
def get_playlist(request: str) -> str:
    """Delegate music and playlist selection to the wedding music specialist."""
    return run_specialist(music_agent, request)


agent = create_agent(
    model=model,
    tools=[get_travel_options, get_venue_options, get_playlist],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are the coordinator of a destination wedding planning team. First extract missing essentials "
        "only when necessary. Delegate travel, venue, and music work to the appropriate specialists. "
        "Combine their findings into a clear plan, separating verified current information from recommendations."
    ),
)
