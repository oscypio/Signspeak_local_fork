package com.example.SignSpeakBackend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class MLResponse {
    private List<MLResult> results;

    public List<MLResult> getResults() { return results; }
    public void setResults(List<MLResult> results) { this.results = results; }

    public static class MLResult {
        private String prediction;
        private String status; // "word_added" oppure "end_of_sentence"

        @JsonProperty("current_words")
        private List<String> currentWords;

        private String sentence;
        private String detail;

        // Getters & Setters
        public String getPrediction() { return prediction; }
        public void setPrediction(String prediction) { this.prediction = prediction; }

        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }

        public List<String> getCurrentWords() { return currentWords; }
        public void setCurrentWords(List<String> currentWords) { this.currentWords = currentWords; }

        public String getSentence() { return sentence; }
        public void setSentence(String sentence) { this.sentence = sentence; }

        public String getDetail() { return detail; }
        public void setDetail(String detail) { this.detail = detail; }
    }
}