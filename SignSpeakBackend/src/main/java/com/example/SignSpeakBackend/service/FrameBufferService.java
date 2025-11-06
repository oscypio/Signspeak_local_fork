package com.example.SignSpeakBackend.service;

import com.example.SignSpeakBackend.model.FrameData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class FrameBufferService {

    private static final Logger logger = LoggerFactory.getLogger(FrameBufferService.class);

    private final ConcurrentLinkedQueue<FrameData> buffer = new ConcurrentLinkedQueue<>();
    private final MLSystemService mlSystemService;
    private final AtomicInteger sequenceCounter = new AtomicInteger(0);

    @Value("${frame.selection.count:30}")
    private int frameSelectionCount;

    public FrameBufferService(MLSystemService mlSystemService) {
        this.mlSystemService = mlSystemService;
    }

    public int getNextSequenceNumber() {
        return sequenceCounter.getAndIncrement();
    }

    public void addFrame(FrameData frameData) {
        buffer.offer(frameData);
        logger.debug("Frame added to buffer. Buffer size: {}", buffer.size());
    }

    @Scheduled(fixedRate = 5000) // Every 5 seconds
    public void processBuffer() {
        if (buffer.isEmpty()) {
            logger.info("Buffer is empty, skipping processing");
            return;
        }

        List<FrameData> allFrames = new ArrayList<>(buffer);
        List<FrameData> selectedFrames = selectDistributedFrames(allFrames, frameSelectionCount);

        logger.info("Processing buffer. Total frames: {}, Selected frames: {}",
                allFrames.size(), selectedFrames.size());

        // Send to ML system
        mlSystemService.sendFrames(selectedFrames);

        // Clear the buffer
        buffer.clear();
    }

    /**
     * Selects frames that are evenly distributed across the time window
     */
    private List<FrameData> selectDistributedFrames(List<FrameData> frames, int count) {
        if (frames.isEmpty()) {
            return new ArrayList<>();
        }

        if (frames.size() <= count) {
            return new ArrayList<>(frames);
        }

        List<FrameData> selected = new ArrayList<>();
        double step = (double) frames.size() / count;

        for (int i = 0; i < count; i++) {
            int index = (int) Math.round(i * step);
            if (index >= frames.size()) {
                index = frames.size() - 1;
            }
            selected.add(frames.get(index));
        }

        return selected;
    }

    public int getBufferSize() {
        return buffer.size();
    }

    public void setFrameSelectionCount(int count) {
        this.frameSelectionCount = count;
        logger.info("Frame selection count updated to: {}", count);
    }

    public int getFrameSelectionCount() {
        return frameSelectionCount;
    }
}