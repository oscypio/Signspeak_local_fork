package com.example.SignSpeakBackend.service;

import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class SimulatedTranslationService {

    private final SimpMessagingTemplate messagingTemplate;

    public SimulatedTranslationService(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }

    public void sendSimulatedTranslation(String meetingId) {
        String destination = "/topic/meeting/" + meetingId;

        try {
            // 1. Parola 1
            Map<String, Object> partial1 = new HashMap<>();
            partial1.put("type", "PARTIAL");
            partial1.put("words", List.of("HELLO"));
            messagingTemplate.convertAndSend(destination, partial1);
            Thread.sleep(800); // Ritardo per effetto visivo

            // 2. Parola 2
            Map<String, Object> partial2 = new HashMap<>();
            partial2.put("type", "PARTIAL");
            partial2.put("words", List.of("HELLO", "FROM"));
            messagingTemplate.convertAndSend(destination, partial2);
            Thread.sleep(800);

            // 3. Parola 3
            Map<String, Object> partial3 = new HashMap<>();
            partial3.put("type", "PARTIAL");
            partial3.put("words", List.of("HELLO", "FROM", "BACKEND"));
            messagingTemplate.convertAndSend(destination, partial3);
            Thread.sleep(800);

            // 4. Frase Finale
            Map<String, Object> finalMsg = new HashMap<>();
            finalMsg.put("type", "FINAL");
            finalMsg.put("text", "Hello from Backend! The connection works.");
            messagingTemplate.convertAndSend(destination, finalMsg);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("Simulated sequence sent to: " + destination);
    }
}