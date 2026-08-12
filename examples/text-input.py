import asyncio

from modihub.llm import LLM
from dotenv import find_dotenv, load_dotenv

async def main() -> None:
    """Generate a text response with an asynchronous model client."""
    load_dotenv(find_dotenv()) # Loads API keys from .env file

    # Replace with your desired model
    llm = await LLM.create("gpt-4o-mini")
    # Generate text
    response = await llm("Tell me a joke about AI.")
    print(response)

asyncio.run(main())
