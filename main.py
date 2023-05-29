import io
import openai
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from similarity import find_closest_sentences, read_sentences_from_csv

load_dotenv(find_dotenv())

# Set the OpenAI API key
openai.api_key = os.environ.get("OPEN_AI_API")

# load model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to("cuda")

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

    # Find the top 5 similar sentences from dataset
    input_sentence = raw_caption
    file_path = "sentences.csv"
    column_name = "captions"

    # function to read sentences from csv file
    sentences = read_sentences_from_csv(file_path, column_name)
    # function to find closest sentences
    top_sentences = find_closest_sentences(input_sentence, sentences, top_k=5)

    # Format the suggestions
    final_suggestions = []
    for sentence in top_sentences:
        final_suggestions.append(sentence)
    formatted_suggestions = "\n".join(
        [f"• {item}" for item in final_suggestions])

    # Return the captions
    return {"captions": formatted_suggestions}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
