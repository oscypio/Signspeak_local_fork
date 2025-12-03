package com.example.SignSpeakBackend.controller;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.service.FrameBufferService;
import com.example.SignSpeakBackend.service.MLSystemService;
import com.example.SignSpeakBackend.service.SimulatedTranslationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class WebSocketController {

    private static final Logger logger = LoggerFactory.getLogger(WebSocketController.class);

    private final FrameBufferService frameBufferService;
    private final MLSystemService mlSystemService;
    private final SimulatedTranslationService simulatedTranslationService;
    private final SimpMessagingTemplate template;

    private final boolean USE_SIMULATION = false;

    public WebSocketController(FrameBufferService frameBufferService,
                               MLSystemService mlSystemService,
                               SimulatedTranslationService simulatedTranslationService,
                               SimpMessagingTemplate template) {
        this.frameBufferService = frameBufferService;
        this.mlSystemService = mlSystemService;
        this.simulatedTranslationService = simulatedTranslationService;
        this.template = template;
    }

    @MessageMapping("/frame")
    public void receiveFrame(FrameData frameData) {
        frameData.setReceivedAt(System.currentTimeMillis());
        frameData.setSequenceNumber(frameBufferService.getNextSequenceNumber());

        String meetingId = "default";
        if (frameData.getUserInfo() != null) {
            meetingId = frameData.getUserInfo().getMeetingId();
        }

        frameBufferService.addFrame(frameData);

        // Checking the size of the buffer
        if (frameBufferService.getBufferSize() >= frameBufferService.getChunkThreshold()) {

            List<FrameData> chunk = frameBufferService.getAndClearBuffer();

            if (USE_SIMULATION) {
                // --- Simulated service ---
                logger.info("Buffer full. Sending SIMULATED response to {}", meetingId);
                simulatedTranslationService.sendSimulatedTranslation(meetingId);
            } else {
                // --- Call the real ML service
                mlSystemService.processFramesAndBroadcast(meetingId, chunk);
            }
        }
    }

    @MessageMapping("/speak")
    public void broadcastAudio(@Payload Map<String, String> payload) {
        try {
            String meetingId = payload.get("meetingId");
            String textToSpeak = payload.get("text");

            if (meetingId != null && textToSpeak != null) {
                Map<String, String> response = new HashMap<>();
                response.put("type", "AUDIO_COMMAND");
                response.put("text", textToSpeak);
                // Aggiungiamo anche meetingId per sicurezza nel filtro frontend
                response.put("meetingId", meetingId);

                String destination = "/topic/meeting/" + meetingId;
                template.convertAndSend(destination, response);
                logger.info("Audio command sent to: {}", destination);
            }
        } catch (Exception e) {
            logger.error("Error in broadcastAudio", e);
        }
    }
}