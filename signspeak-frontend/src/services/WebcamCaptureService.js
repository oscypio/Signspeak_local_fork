/**
 * Requests access to the user's webcam and returns the MediaStream.
 * It also handles authorization or device errors.
 */
export const getWebcamStream = async () => {
    try {
        // Request video-only access from the browser
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false, // We don't need audio for landmark analysis
        });
        return stream;

    } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            console.error("User denied webcam access.");
            alert("Please allow webcam access to use the application.");
        } else {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please ensure it's not used by another app.");
        }
        return null; // Return null on error
    }
};