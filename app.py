from flask import Flask, render_template, request
import os

from inference import generate_caption


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    caption = None
    image_path = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."

            return render_template(
                "index.html",
                caption=caption,
                image_path=image_path,
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."

            return render_template(
                "index.html",
                caption=caption,
                image_path=image_path,
                error=error
            )

        # Save uploaded image
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(image_path)

        # Generate caption
        try:

            caption = generate_caption(
                image_path
            )

        except Exception as e:

            error = f"Error generating caption: {str(e)}"

    return render_template(
        "index.html",
        caption=caption,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)