import streamlit as st
import torch
import openai
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import requests

# Object creation model, tokenizer, and processor from HuggingFace
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prediction(img_list):
    """
    Generates captions for a list of images using the BLIP model.
    """
    max_length = 50
    num_beams = 5
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    img = []
    for image in tqdm(img_list):
        i_image = Image.open(image)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        img.append(i_image)

    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    output = model.generate(pixel_val, **gen_kwargs)

    predict = processor.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict

def openai_completion(prompt, api_key):
    """
    Generates a completion for a given prompt using the OpenAI API.
    """
    openai.api_key = api_key  # Set the API key
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # Provide the model argument
        prompt=prompt,
        max_tokens=30,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

def generate_image(description, api_key):
    """
    Generates an image based on a given description using the OpenAI API.
    """
    url = 'https://api.openai.com/v1/images/generations'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        "prompt": description,
        "n": 1,
        "size": "1024x1024"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        json_data = response.json()
        return json_data['data'][0]['url']
    else:
        st.error(f"Error generating image with status code {response.status_code}")
        return None

def main():
    # title on the tab
    st.set_page_config(page_title="Fashion Inspiration")

    # Title of the page
    st.title("Get Fashion Inspiration for your Image By DeepSuthar")
    # Text input for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # Upload images section
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    # Checkbox to generate image
    generate_image_flag = st.checkbox("Generate Ai Image and Suggestion")

    # Submit button
    if st.button("Submit"):
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(Image.open(uploaded_file), width=300)

            # Generate a caption for the image
            captions = prediction([uploaded_file])
            st.subheader("Image Caption")
            st.write(captions[0])

            if openai_api_key and generate_image_flag:
                # Use OpenAI to generate a fashion variation idea based on the caption
                prompt = f"Generate a fashion variation idea based on this description and please note that the output token length is 30: {captions[0]}"
                fashion_idea = openai_completion(prompt, openai_api_key)
                st.subheader("Fashion Variation Idea")
                st.write(fashion_idea)

                # Generate an image based on the fashion variation idea
                st.subheader("Generated Image based on Fashion Variation Idea")
                image_url = generate_image(fashion_idea, openai_api_key)
                if image_url is not None:
                    st.image(image_url)
                else:
                    st.warning("Image generation failed. Please check your OpenAI API Key.")
            elif not openai_api_key:
                st.warning("Please enter your OpenAI API Key to generate Fashion Variation Idea and Image.")
            elif not generate_image_flag:
                st.warning("Image and text generation is disabled by you so that image and text is not genreted ✌.")

if __name__ == "__main__":
    main()
