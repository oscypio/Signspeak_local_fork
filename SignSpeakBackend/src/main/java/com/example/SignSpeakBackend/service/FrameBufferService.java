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

    private final ConcurrentLinkedQueue<FrameData> buffer = new ConcurrentLinkedQueue<>();
    private final AtomicInteger sequenceCounter = new AtomicInteger(0);

    private int chunkThreshold = 10;

    public int getNextSequenceNumber() {
        return sequenceCounter.getAndIncrement();
    }

    public void addFrame(FrameData frameData) {
        buffer.offer(frameData);
    }

    public int getBufferSize() {
        return buffer.size();
    }

    public int getChunkThreshold() {
        return chunkThreshold;
    }

    public void setChunkThreshold(int chunkThreshold) {
        this.chunkThreshold = chunkThreshold;
    }

    /**
     * Thread-safe.
     */
    public List<FrameData> getAndClearBuffer() {
        List<FrameData> frames = new ArrayList<>();
        FrameData frame;
        while ((frame = buffer.poll()) != null) {
            frames.add(frame);
        }
        return frames;
    }

    public void clearBuffer() {
        this.buffer.clear();
    }
}