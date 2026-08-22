from flask import Flask, render_template, request
from model import generate_caption
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return "No image uploaded"

    image = request.files["image"]

    if image.filename == "":
        return "No image selected"

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    caption = generate_caption(image_path)

    return render_template(
        "index.html",
        caption=caption,
        image=image.filename
    )


if __name__ == "__main__":
    app.run(debug=True)