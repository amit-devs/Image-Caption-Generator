// ============================================================
// ELEMENTS
// ============================================================

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const uploadContent = document.getElementById("uploadContent");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const fileName =
    document.getElementById("fileName");

const removeImage =
    document.getElementById("removeImage");

const uploadForm =
    document.getElementById("uploadForm");

const generateBtn =
    document.getElementById("generateBtn");

const details =
    document.getElementById("details");

const charCount =
    document.querySelector(".char-count");


// ============================================================
// IMAGE PREVIEW
// ============================================================

function showPreview(file) {

    if (!file) {
        return;
    }

    // Check whether selected file is an image
    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image file.");

        return;
    }


    // Show filename

    fileName.textContent = file.name;


    // Read image

    const reader = new FileReader();


    reader.onload = function (event) {

        imagePreview.src =
            event.target.result;

        previewContainer.classList.add(
            "active"
        );

        uploadContent.style.opacity = "0";

    };


    reader.readAsDataURL(file);
}


// ============================================================
// FILE INPUT
// ============================================================

if (imageInput) {

    imageInput.addEventListener(
        "change",
        function () {

            if (this.files.length > 0) {

                showPreview(
                    this.files[0]
                );

            }

        }
    );

}


// ============================================================
// DROP ZONE
// ============================================================

if (dropZone) {


    // Drag over

    dropZone.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            dropZone.classList.add(
                "dragover"
            );

        }
    );


    // Drag leave

    dropZone.addEventListener(
        "dragleave",
        function () {

            dropZone.classList.remove(
                "dragover"
            );

        }
    );


    // Drop

    dropZone.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            dropZone.classList.remove(
                "dragover"
            );


            const files =
                event.dataTransfer.files;


            if (files.length > 0) {

                const file = files[0];


                // Put dropped file into input

                try {

                    const dataTransfer =
                        new DataTransfer();

                    dataTransfer.items.add(
                        file
                    );

                    imageInput.files =
                        dataTransfer.files;

                } catch (error) {

                    console.log(
                        "Could not assign dropped file."
                    );

                }


                showPreview(file);
            }

        }
    );


    // Click anywhere on drop zone

    dropZone.addEventListener(
        "click",
        function (event) {

            // Don't open file picker
            // when clicking remove button

            if (
                removeImage &&
                removeImage.contains(event.target)
            ) {

                return;

            }


            imageInput.click();

        }
    );

}


// ============================================================
// REMOVE IMAGE
// ============================================================

if (removeImage) {

    removeImage.addEventListener(
        "click",
        function (event) {

            event.preventDefault();

            event.stopPropagation();


            // Clear input

            imageInput.value = "";


            // Clear preview

            imagePreview.src = "";


            // Hide preview

            previewContainer.classList.remove(
                "active"
            );


            // Restore upload content

            uploadContent.style.opacity =
                "1";


            // Reset filename

            fileName.textContent =
                "No image selected";

        }
    );

}


// ============================================================
// GENERATE BUTTON LOADING
// ============================================================

if (uploadForm && generateBtn) {

    uploadForm.addEventListener(
        "submit",
        function () {

            // Make sure an image exists

            if (
                !imageInput.files ||
                imageInput.files.length === 0
            ) {

                return;

            }


            // Disable button

            generateBtn.disabled = true;


            // Loading animation

            generateBtn.innerHTML = `

                <span class="loading-spinner"></span>

                <span>
                    Analyzing Image...
                </span>

            `;

        }
    );

}


// ============================================================
// CHARACTER COUNTER
// ============================================================

if (details && charCount) {

    details.addEventListener(
        "input",
        function () {

            const length =
                this.value.length;


            charCount.textContent =
                `${length} / 300`;


            // Warning when approaching limit

            if (length >= 270) {

                charCount.style.color =
                    "#f59e0b";

            }


            if (length >= 295) {

                charCount.style.color =
                    "#ef4444";

            }


            if (length < 270) {

                charCount.style.color =
                    "#71717a";

            }

        }
    );

}


// ============================================================
// REGENERATE BUTTON ANIMATION
// ============================================================

const regenerateForm =
    document.querySelector(
        ".regenerate-form"
    );

const regenerateBtn =
    document.querySelector(
        ".regenerate-btn"
    );


if (
    regenerateForm &&
    regenerateBtn
) {

    regenerateForm.addEventListener(
        "submit",
        function () {

            regenerateBtn.disabled = true;


            regenerateBtn.innerHTML = `

                <span class="loading-spinner small"></span>

                Generating...

            `;

        }
    );

}


// ============================================================
// IMPROVE BUTTON ANIMATION
// ============================================================

const improveForm =
    document.querySelector(
        ".improve-content form"
    );

const improveBtn =
    document.querySelector(
        ".improve-btn"
    );


if (
    improveForm &&
    improveBtn
) {

    improveForm.addEventListener(
        "submit",
        function () {

            improveBtn.disabled = true;


            improveBtn.innerHTML = `

                <span class="loading-spinner small"></span>

                Improving...

            `;

        }
    );

}


// ============================================================
// ADD LOADING SPINNER STYLE
// ============================================================

const spinnerStyle =
    document.createElement("style");


spinnerStyle.textContent = `

    .loading-spinner {

        width: 18px;
        height: 18px;

        border: 2px solid
            rgba(255, 255, 255, 0.35);

        border-top-color: white;

        border-radius: 50%;

        display: inline-block;

        animation:
            spin 0.7s linear infinite;

    }


    .loading-spinner.small {

        width: 15px;
        height: 15px;

    }


    button:disabled {

        opacity: 0.75;

        cursor: wait;

        transform: none !important;

    }


    @keyframes spin {

        to {

            transform:
                rotate(360deg);

        }

    }

`;


document.head.appendChild(
    spinnerStyle
);