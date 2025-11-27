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
     * @param {string} meetingId - The ID of the meeting to subscribe to.
     */
    connect(callbacks = {}, meetingId) {
        const { onOpen = () => {}, onMessage = () => {}, onError = () => {}, onClose = () => {} } = callbacks;

        try {
            // Nota: Per il deploy, assicurati che questo URL non sia hardcodato su localhost
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
                    // console.log('STOMP: ' + str); // Decommenta per debug profondo
                },
                reconnectDelay: 5000, // È meglio avere un riconnessione automatica
                heartbeatIncoming: 4000,
                heartbeatOutgoing: 4000,
            });

            this.stompClient.onConnect = () => {
                console.log('STOMP: Connected to WebSocket');
                onOpen();

                // 1. Definisci il topic dinamico basato sull'ID del meeting
                // Assicurati che il backend supporti questo pattern di URL
                const subscriptionTopic = `/topic/meeting/${meetingId}`;

                console.log(`STOMP: Subscribing to: ${subscriptionTopic}`);

                // 2. CORREZIONE: Usa la variabile subscriptionTopic qui!
                this.stompClient.subscribe(subscriptionTopic, (message) => {
                    console.log('STOMP: Message received from topic:', subscriptionTopic);
                    onMessage(message.body);
                });
            };

            this.stompClient.onStompError = (frame) => {
                console.error('STOMP Error: ' + frame.headers['message']);
                console.error('STOMP Details: ' + frame.body);
                onError(frame);
            };

            this.stompClient.onDisconnect = (frame) => {
                console.log('STOMP: Disconnected');
                onClose(frame);
            };

            this.stompClient.activate();
        } catch (error){
            console.error("Failed to initialize connection:", error);
            onError(error);
        }
    }

    /**
     * Sends hand landmark data to the backend.
     */
    sendHandData(results, timestamp, userInfo) {
        if (!this.stompClient || !this.stompClient.connected) {
            // console.warn("STOMP client is not connected."); // Evita spam in console se disconnesso
            return;
        }

        const dataPacket = {
            timestamp: timestamp,
            landmarks: results.landmarks,
            handedness: results.handedness,
            // Importante: userInfo contiene meetingId e userStatus (DEAF/NORMAL)
            userInfo: userInfo
        };

        // Invia i dati al backend.
        // Il backend leggerà userInfo.meetingId per sapere su quale topic rispedire la risposta.
        this.stompClient.publish({
            destination: '/app/frame',
            body: JSON.stringify(dataPacket)
        });
    }

    disconnect() {
        if (this.stompClient && this.stompClient.connected) {
            this.stompClient.deactivate();
            console.log("STOMP client deactivated.");
        }
    }
}