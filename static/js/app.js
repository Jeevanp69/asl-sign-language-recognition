console.log("ASL Vision frontend loaded.");

const predictionLetter =
    document.querySelector(".prediction-letter");

const confidenceValue =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidence-bar");

const predictionMessage =
    document.querySelector(".prediction-message strong");

const predictionDescription =
    document.querySelector(".prediction-message p");


async function updatePrediction() {

    try {

        const response =
            await fetch("/prediction");

        const data =
            await response.json();


        // ----------------------------------------------------
        // Current letter
        // ----------------------------------------------------

        if (data.letter) {

            predictionLetter.textContent =
                data.letter;

        } else {

            predictionLetter.textContent =
                "—";
        }


        // ----------------------------------------------------
        // Confidence
        // ----------------------------------------------------

        confidenceValue.textContent =
            `${data.confidence}%`;

        confidenceBar.style.width =
            `${data.confidence}%`;


        // ----------------------------------------------------
        // Status
        // ----------------------------------------------------

        if (data.status === "Detected") {

            predictionMessage.textContent =
                "Letter detected";

            predictionDescription.textContent =
                `The model detected ${data.letter} ` +
                `with ${data.confidence}% confidence.`;

        } else if (
            data.status === "Low confidence"
        ) {

            predictionMessage.textContent =
                "Low confidence";

            predictionDescription.textContent =
                "Try holding your hand steady " +
                "and make the gesture clearer.";

        } else {

            predictionMessage.textContent =
                "Ready for detection";

            predictionDescription.textContent =
                "Hold a clear ASL alphabet gesture " +
                "in front of the camera.";
        }


        // ----------------------------------------------------
        // Recognized text
        // ----------------------------------------------------

        const textBox =
            document.getElementById("recognized-text");

        if (textBox) {

            textBox.textContent =
                data.text || "Your recognized text will appear here...";
        }

    } catch (error) {

        console.error(
            "Prediction request failed:",
            error
        );
    }
}


async function sendAction(endpoint) {

    try {

        const response =
            await fetch(
                endpoint,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        const textBox =
            document.getElementById("recognized-text");

        if (textBox) {

            textBox.textContent =
                data.text || "Your recognized text will appear here...";
        }

    } catch (error) {

        console.error(
            "Action failed:",
            error
        );
    }
}


// ------------------------------------------------------------
// Buttons
// ------------------------------------------------------------

const clearButton =
    document.getElementById("clear-text");

if (clearButton) {

    clearButton.addEventListener(
        "click",
        () => sendAction("/clear_text")
    );
}


const spaceButton =
    document.getElementById("space-text");

if (spaceButton) {

    spaceButton.addEventListener(
        "click",
        () => sendAction("/space")
    );
}


const backspaceButton =
    document.getElementById("backspace-text");

if (backspaceButton) {

    backspaceButton.addEventListener(
        "click",
        () => sendAction("/backspace")
    );
}


// Update prediction five times per second
setInterval(
    updatePrediction,
    200
);

updatePrediction();

// ------------------------------------------------------------
// Speak recognized text
// ------------------------------------------------------------

const speakButton =
    document.getElementById("speak-text");


if (speakButton) {

    speakButton.addEventListener(
        "click",
        async () => {

            const textBox =
                document.getElementById(
                    "recognized-text"
                );

            const text =
                textBox.textContent.trim();


            if (
                !text ||
                text ===
                "Your recognized text will appear here..."
            ) {

                return;
            }


            try {

                speakButton.disabled = true;

                speakButton.textContent =
                    "🔊 Speaking...";


                await fetch(
                    "/speak",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            text: text
                        })
                    }
                );

            } catch (error) {

                console.error(
                    "Speech request failed:",
                    error
                );

            } finally {

                speakButton.disabled = false;

                speakButton.textContent =
                    "🔊 Speak Text";
            }
        }
    );
}