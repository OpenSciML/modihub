import asyncio

from PIL import Image
from dotenv import find_dotenv, load_dotenv

from modihub.llm import LLM

load_dotenv(find_dotenv())

async def main() -> None:
    """List models and generate a multimodal response asynchronously."""
    available_models = await LLM.available_models()
    for client, models in available_models.group_by("client"):
        for model in models:
            print(f"{client}: {model.name}")

    for model in available_models.filter_by("client", "openai"):
        print(model)

    llm = await LLM.create("models/gemini-3.5-flash")
    image = Image.open("image.png")
    text = "Describe the following image"
    prompt = [text, image]
    response = await llm(prompt)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
