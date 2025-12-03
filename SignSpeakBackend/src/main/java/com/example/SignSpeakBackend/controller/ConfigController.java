package com.example.SignSpeakBackend.controller;

import com.example.SignSpeakBackend.service.FrameBufferService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ConfigController {
    
    private final FrameBufferService frameBufferService;

    public ConfigController(FrameBufferService frameBufferService) {
        this.frameBufferService = frameBufferService;
    }

    @GetMapping("/config")
    public ResponseEntity<Map<String, Object>> getConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("frameSelectionCount", frameBufferService.getChunkThreshold());
        config.put("bufferSize", frameBufferService.getBufferSize());
        return ResponseEntity.ok(config);
    }

    @PostMapping("/config/frame-count")
    public ResponseEntity<Map<String, Object>> updateFrameCount(@RequestBody Map<String, Integer> request) {
        Integer count = request.get("count");
        if (count != null && count > 0) {
            frameBufferService.setChunkThreshold(count);
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("frameSelectionCount", count);
            return ResponseEntity.ok(response);
        }
        return ResponseEntity.badRequest().body(Map.of("success", false, "error", "Invalid count"));
    }
}