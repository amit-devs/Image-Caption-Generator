from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the pre-trained BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    output = model.generate(**inputs, max_new_tokens=50)

    # Convert generated tokens into text
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption