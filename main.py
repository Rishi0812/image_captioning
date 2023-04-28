import io
import openai
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Set the OpenAI API key
openai.api_key = os.environ.get("OPEN_AI_API")

#load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# Create the FastAPI app
app = FastAPI()

# Define the root endpoint with the HTML response
@app.get("/", response_class=HTMLResponse)
async def root():
    return open("static/index.html").read()

# Define the API endpoint for generating captions from uploaded image file
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    image_bytes = await file.read()

    raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    raw_caption = processor.decode(out[0], skip_special_tokens=True)

    # instagram worthy caption using openai
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can help give instagram worthy attractive captions based on the image description. Your output is always in proper format with each caption in seperate pointers (•) - followed by relevant hashtags\n\n"},
                {
                    "role": "system",
                    "content": f"Give me 10 attractive instagram captions for the image having: {raw_caption}\n",
                }
            ],
            temperature=0.2
        )
    captions_text = completion["choices"][0]["message"]["content"]
    captions_text = captions_text.replace("•", "\n•")

    # Return the captions
    return {"captions": captions_text}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


