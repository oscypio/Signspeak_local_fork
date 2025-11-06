package com.example.SignSpeakBackend.controller;

import com.example.SignSpeakBackend.model.FrameData;
import com.example.SignSpeakBackend.service.FrameBufferService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.stereotype.Controller;

@Controller
public class WebSocketController {

    private static final Logger logger = LoggerFactory.getLogger(WebSocketController.class);

    private final FrameBufferService frameBufferService;

    public WebSocketController(FrameBufferService frameBufferService) {
        this.frameBufferService = frameBufferService;
    }

    @MessageMapping("/frame")
    @SendTo("/topic/status")
    public String receiveFrame(FrameData frameData) {
        // Add server-side metadata
        frameData.setReceivedAt(System.currentTimeMillis());
        frameData.setSequenceNumber(frameBufferService.getNextSequenceNumber());

        logger.debug("Received frame seq#{} with {} hand(s) detected (timestamp: {}ms)",
                frameData.getSequenceNumber(),
                frameData.getHandCount(),
                String.format("%.2f", frameData.getTimestamp()));

        frameBufferService.addFrame(frameData);

        return String.format("Frame received. Buffer: %d frames", frameBufferService.getBufferSize());
    }
}