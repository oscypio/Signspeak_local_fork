package com.example.SignSpeakBackend.service;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.model.MLResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MLSystemService {

    private static final Logger logger = LoggerFactory.getLogger(MLSystemService.class);

    // URL indicato nel README del ML (assicurati che la porta sia giusta, es. 8000)
    @Value("${ml.system.url:http://localhost:8000/api/predict_landmarks}")
    private String mlSystemUrl;

    private final RestTemplate restTemplate;
    private final SimpMessagingTemplate messagingTemplate;

    public MLSystemService(RestTemplate restTemplate, SimpMessagingTemplate messagingTemplate) {
        this.restTemplate = restTemplate;
        this.messagingTemplate = messagingTemplate;
    }

    /**
     * Invia un chunk di frame al ML e gestisce la risposta.
     */
    public void processFramesAndBroadcast(String meetingId, List<FrameData> frames) {
        if (frames.isEmpty()) return;

        try {
            // logger.info("Sending {} frames to ML for meeting {}", frames.size(), meetingId);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<List<FrameData>> request = new HttpEntity<>(frames, headers);

            // Chiamata POST all'API Python
            MLResponse response = restTemplate.postForObject(mlSystemUrl, request, MLResponse.class);

            if (response != null && response.getResults() != null) {
                for (MLResponse.MLResult result : response.getResults()) {
                    broadcastResult(meetingId, result);
                }
            }

        } catch (Exception e) {
            logger.error("Error communicating with ML system: {}", e.getMessage());
            // Opzionale: Inviare un errore al frontend se serve
        }
    }

    private void broadcastResult(String meetingId, MLResponse.MLResult result) {
        String destination = "/topic/meeting/" + meetingId;
        Map<String, Object> message = new HashMap<>();

        if ("word_added".equals(result.getStatus())) {
            // Caso 1: Parola rilevata -> Aggiornamento parziale
            message.put("type", "PARTIAL");
            message.put("words", result.getCurrentWords()); // Es. ["NEED", "PHONE"]
            message.put("last_prediction", result.getPrediction());

            logger.info("Broadcasting PARTIAL: {} to {}", result.getCurrentWords(), destination);

        } else if ("end_of_sentence".equals(result.getStatus())) {
            // Caso 2: Frase finita -> Sostituzione finale
            message.put("type", "FINAL");
            message.put("text", result.getSentence()); // Es. "I need a phone."

            logger.info("Broadcasting FINAL: {} to {}", result.getSentence(), destination);
        }

        if (!message.isEmpty()) {
            messagingTemplate.convertAndSend(destination, message);
        }
    }

    public void resetMLContext() {
        try {
            String url = mlSystemUrl.replace("/predict_landmarks", "/reset_buffer");
            restTemplate.postForObject(url, null, String.class);
            logger.info("Sent RESET command to ML Service");
        } catch (Exception e) {
            logger.error("Failed to reset ML Service: {}", e.getMessage());

        }
    }
}