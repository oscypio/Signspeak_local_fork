package com.example.SignSpeakBackend.model;

import java.util.List;

public class FrameData {
    // Performance timestamp from frontend (ms since page load)
    private double timestamp;

    // Server-side metadata
    private Long receivedAt; // System timestamp when server receives the frame
    private Integer sequenceNumber; // Server-assigned sequence number

    // MediaPipe Hand Landmarker results
    // Each inner array represents one detected hand (21 landmarks per hand)
    private List<List<Landmark>> landmarks;

    // Handedness info for each detected hand
    private List<List<HandInfo>> handedness;

    public FrameData() {}

    public double getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(double timestamp) {
        this.timestamp = timestamp;
    }

    public Long getReceivedAt() {
        return receivedAt;
    }

    public void setReceivedAt(Long receivedAt) {
        this.receivedAt = receivedAt;
    }

    public Integer getSequenceNumber() {
        return sequenceNumber;
    }

    public void setSequenceNumber(Integer sequenceNumber) {
        this.sequenceNumber = sequenceNumber;
    }

    public List<List<Landmark>> getLandmarks() {
        return landmarks;
    }

    public void setLandmarks(List<List<Landmark>> landmarks) {
        this.landmarks = landmarks;
    }

    public List<List<HandInfo>> getHandedness() {
        return handedness;
    }

    public void setHandedness(List<List<HandInfo>> handedness) {
        this.handedness = handedness;
    }


    public int getHandCount() {
        return landmarks != null ? landmarks.size() : 0;
    }

    public static class Landmark {
        private double x;
        private double y;
        private double z;
        private double visibility;

        public Landmark() {}

        public Landmark(double x, double y, double z, double visibility) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.visibility = visibility;
        }

        public double getX() {
            return x;
        }

        public void setX(double x) {
            this.x = x;
        }

        public double getY() {
            return y;
        }

        public void setY(double y) {
            this.y = y;
        }

        public double getZ() {
            return z;
        }

        public void setZ(double z) {
            this.z = z;
        }

        public double getVisibility() {
            return visibility;
        }

        public void setVisibility(double visibility) {
            this.visibility = visibility;
        }
    }

    public static class HandInfo {
        private double score;        // Confidence score (0-1)
        private int index;           // Hand index (0 or 1)
        private String categoryName; // "Left" or "Right"
        private String displayName;  // Display name (usually same as categoryName)

        public HandInfo() {}

        public double getScore() {
            return score;
        }

        public void setScore(double score) {
            this.score = score;
        }

        public int getIndex() {
            return index;
        }

        public void setIndex(int index) {
            this.index = index;
        }

        public String getCategoryName() {
            return categoryName;
        }

        public void setCategoryName(String categoryName) {
            this.categoryName = categoryName;
        }

        public String getDisplayName() {
            return displayName;
        }

        public void setDisplayName(String displayName) {
            this.displayName = displayName;
        }
    }
}