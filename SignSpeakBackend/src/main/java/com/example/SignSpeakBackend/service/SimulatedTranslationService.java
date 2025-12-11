package com.example.SignSpeakBackend.service;

import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

@Service
public class SimulatedTranslationService {

    private final SimpMessagingTemplate messagingTemplate;

    private final Map<String, Future<?>> activeSimulations = new ConcurrentHashMap<>();
    private final ExecutorService executorService = Executors.newCachedThreadPool();

    public SimulatedTranslationService(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }

    /**
     * Interrompe qualsiasi simulazione attiva per questo meetingId
     */
    public void stopSimulation(String meetingId) {
        Future<?> future = activeSimulations.remove(meetingId);
        if (future != null && !future.isDone()) {
            future.cancel(true);
            System.out.println("Simulation STOPPED for meeting: " + meetingId);
        }
    }

    public void sendSimulatedTranslation(String meetingId) {
        stopSimulation(meetingId);

        Future<?> future = executorService.submit(() -> {
            String destination = "/topic/meeting/" + meetingId;
            try {
                if (Thread.currentThread().isInterrupted()) return;
                sendPartial(destination, List.of("HELLO"));
                Thread.sleep(800);

                if (Thread.currentThread().isInterrupted()) return;
                sendPartial(destination, List.of("HELLO", "FROM"));
                Thread.sleep(800);

                if (Thread.currentThread().isInterrupted()) return;
                sendPartial(destination, List.of("HELLO", "FROM", "BACKEND"));
                Thread.sleep(800);

                if (Thread.currentThread().isInterrupted()) return;
                sendFinal(destination, "Hello from Backend! The connection works.");

            } catch (InterruptedException e) {
                System.out.println("Simulation INTERRUPTED via CLEAR command.");
                Thread.currentThread().interrupt();
            } finally {
                activeSimulations.remove(meetingId);
            }
        });

        activeSimulations.put(meetingId, future);
    }

    private void sendPartial(String destination, List<String> words) {
        Map<String, Object> msg = new HashMap<>();
        msg.put("type", "PARTIAL");
        msg.put("words", words);
        messagingTemplate.convertAndSend(destination, msg);
    }

    private void sendFinal(String destination, String text) {
        Map<String, Object> msg = new HashMap<>();
        msg.put("type", "FINAL");
        msg.put("text", text);
        messagingTemplate.convertAndSend(destination, msg);
    }
}