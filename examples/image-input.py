import asyncio

from PIL import Image
from modihub.llm import LLM
from dotenv import find_dotenv, load_dotenv

async def main() -> None:
    """Generate an image description with an asynchronous multimodal client."""
    load_dotenv(find_dotenv())

    llm = await LLM.create("models/gemini-1.5-flash-8b")
    image = Image.open("image.png")  # Replace with the path to your image
    text = "Describe the following image"
    prompt = [text, image]
    response = await llm(prompt)
    print(response)

asyncio.run(main())
