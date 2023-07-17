# Automatic Instagram Ready Image Captioning with image input

This project provides an automatic image captioning system using a pre-trained image captioning model. It generates captions for images uploaded through a FastAPI-based web service. Additionally, it suggests similar sentences from a dataset based on the generated caption.

## Code Structure

The project consists of two Python files:

### `main.py`

This file contains the main code for the FastAPI web service and the image captioning functionality.

#### Dependencies

- `io`
- `fastapi`
- `fastapi.responses`
- `transformers`
- `PIL`
- `similarity` (custom module)

#### Functions and Endpoints

- `root()`: Root endpoint that returns an HTML response with the contents of the `static/index.html` file.
- `/caption` (POST endpoint): Accepts an uploaded image file and generates captions for the image. It uses the pre-trained image captioning model to generate the caption and finds the top 5 similar sentences from a dataset based on the generated caption.

#### Running the App

The app can be run using the command `uvicorn main:app --host 0.0.0.0 --port 8080`.

### `similarity.py`

This file contains helper functions for finding the closest sentences to a given input sentence from a dataset.

#### Dependencies

- `spacy`
- `csv`
- `sklearn.metrics.pairwise`

#### Functions

- `find_closest_sentences(input_sentence, sentences, top_k=5)`: Finds the closest sentences to the given input sentence from a list of sentences using cosine similarity. It returns the top `k` closest sentences.
- `read_sentences_from_csv(file_path, column_name)`: Reads sentences from a CSV file. It returns a list of sentences from the specified column.

## Usage

To use the image captioning system:

1. Install the required dependencies listed in `main.py` using `pip`.
2. Download the pre-trained image captioning model by Salesforce: `"Salesforce/blip-image-captioning-base"`.
3. Place the `static/index.html` file in the appropriate location.
4. Run the `main.py` script using the provided command.
5. Access the web service by opening the root endpoint in a web browser.
6. Upload an image file to the `/caption` endpoint to generate captions.

## Alternative Option

If desired, you can utilize an OpenAI API or any other Language Model (LLM) instead of the dataset similarity approach. By using an LLM, you can generate caption suggestions directly from the model instead of relying on a predefined dataset. You would need to modify the code accordingly to integrate with the chosen LLM.

Please note that using an LLM may require additional configuration and API access credentials. Refer to the documentation of the specific LLM or OpenAI API for instructions on how to integrate it into the project.

---

This README provides an overview of the Automatic Instagram Ready Image Captioning project and instructions on how to set it up and use it. Feel free to customize it further based on your specific project requirements.
