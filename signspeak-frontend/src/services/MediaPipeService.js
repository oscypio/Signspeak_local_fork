import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export class MediaPipeService {
    handLandmarker = null;
    animationFrameId = null;
    lastVideoTime = -1;
    results = null;
    onResultsCallback = null;

    /**
     * Initializes the HandLandmarker model.
     * @param {function} onResults - The callback function to handle landmark results.
     */
    async initialize(onResults) {
        this.onResultsCallback = onResults;

        const vision = await FilesetResolver.forVisionTasks(
            // Use the CDN to load model files
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        // Create the HandLandmarker instance
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                // Use the 'lite' model for better performance
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 2,
        });

        console.log("MediaPipe HandLandmarker initialized.");
    }

    /**
     * Starts the landmark detection loop.
     * @param {HTMLVideoElement} videoElement - The <video> element with the webcam feed.
     */
    startProcessing(videoElement) {
        if (!this.handLandmarker) {
            console.error("MediaPipe service is not initialized.");
            return;
        }

        // Start the prediction loop
        this.predictWebcam(videoElement);
    }

    /**
     * The main prediction loop, running on every animation frame.
     */
    predictWebcam = (videoElement) => {
        const videoTime = videoElement.currentTime;
        const timestamp = performance.now();

        // Process the frame if it's new
        if (this.lastVideoTime !== videoTime) {
            this.lastVideoTime = videoTime;
            // Perform landmark detection
            this.results = this.handLandmarker.detectForVideo(videoElement, timestamp);

            // Send the landmarks back to the App component
            if (this.results.landmarks && this.results.landmarks.length > 0) {
                this.onResultsCallback(this.results, timestamp);
            }
        }

        // Continue the loop
        this.animationFrameId = requestAnimationFrame(() => this.predictWebcam(videoElement));
    };

    /**
     * Stops the landmark detection loop.
     */
    stopProcessing() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        console.log("MediaPipe processing stopped.");
    }
}