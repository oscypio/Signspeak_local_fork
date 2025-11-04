export class WebSocketManager {
    socket = null;
    onMessageCallback = null;

    /**
     * Establishes a WebSocket connection to the given URL.
     * @param {string} url - The backend WebSocket URL.
     * @param {function} onMessage - Callback function for received messages.
     */
    connect(url, onMessage) {
        // Ensure no existing connection
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            console.log("WebSocket is already connected.");
            return;
        }

        this.socket = new WebSocket(url);
        this.onMessageCallback = onMessage;

        this.socket.onopen = () => {
            console.log("WebSocket connection established.");
        };

        this.socket.onmessage = (event) => {
            // Pass the received data to the callback function
            if (this.onMessageCallback) {
                this.onMessageCallback(event.data);
            }
        };

        this.socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        this.socket.onclose = () => {
            console.log("WebSocket connection closed.");
        };
    }

    /**
     * Sends formatted hand data (landmarks, handedness, timestamp) to the backend.
     */
    sendHandData(results, timestamp) {
        const dataPacket = {
            timestamp: timestamp,
            landmarks: results.landmarks,
            handedness: results.handedness
        };

        console.log("Sending data to backend:", JSON.stringify(dataPacket));

        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            // Convert the landmark data to a JSON string before sending
            this.socket.send(JSON.stringify(dataPacket));
        } else {
            console.warn("WebSocket is not open. Cannot send landmarks.");
        }
    }

    /**
     * Closes the WebSocket connection.
     */
    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }
}