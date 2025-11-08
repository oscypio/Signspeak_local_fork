import { Client } from '@stomp/stompjs';
import SockJS from 'sockjs-client';

/**
 * Manages the STOMP connection over SockJS to the backend.
 */
export class WebSocketManager {
    stompClient = null;

    /**
     * Establishes a STOMP connection.
     * @param {object} callbacks - An object with { onOpen, onMessage, onError, onClose }
     */
    connect(callbacks = {}) {
        // Provide default empty functions for callbacks
        const { onOpen = () => {}, onMessage = () => {}, onError = () => {}, onClose = () => {} } = callbacks;
        try {
            const socket = new SockJS('http://localhost:8080/ws');

            socket.onclose = (event) => {
                console.error('SockJS connection closed:', event.reason || 'Cannot connect');
                onClose(event);
            };

            socket.onerror = (error) => {
                console.error('SockJS error:', error);
                onError(error);
            };

            this.stompClient = new Client({
                webSocketFactory: () => socket,
                debug: (str) => {
                    console.log('STOMP: ' + str);
                },
                reconnectDelay: 0,
                heartbeatIncoming: 0,
                heartbeatOutgoing: 0,
            });

            // Handle the successful connection event
            this.stompClient.onConnect = () => {
                console.log('STOMP: Connected to WebSocket');
                onOpen();

                // Subscribe to the topic to receive translations
                this.stompClient.subscribe('/topic/status', (message) => {
                    console.log('STOMP: Message received:', message.body);
                    onMessage(message.body);
                });
            };

            // Handle STOMP errors
            this.stompClient.onStompError = (frame) => {
                console.error('STOMP Error: ' + frame.headers['message']);
                console.error('STOMP Details: ' + frame.body);
                onError(frame);
            };

            // Handle disconnection
            this.stompClient.onDisconnect = (frame) => {
                console.log('STOMP: Disconnected');
                onClose(frame);
            };

            // Activate the client to start the connection
            this.stompClient.activate();
        } catch (error){
            console.error("Failed to initialize connection:", error);
            onError(error);
        }
    }

    /**
     * Sends hand landmark data to the backend.
     */
    sendHandData(results, timestamp) {
        if (!this.stompClient || !this.stompClient.connected) {
            console.warn("STOMP client is not connected. Cannot send data.");
            return;
        }

        const dataPacket = {
            timestamp: timestamp,
            landmarks: results.landmarks,
            handedness: results.handedness
        };

        // Publish the data to the correct destination
        this.stompClient.publish({
            destination: '/app/frame',
            body: JSON.stringify(dataPacket)
        });
    }

    /**
     * Deactivates the STOMP client (closes the connection).
     */
    disconnect() {
        if (this.stompClient && this.stompClient.connected) {
            this.stompClient.deactivate();
        } else {
            console.log("STOMP client already disconnected or not initialized.");
        }
    }
}