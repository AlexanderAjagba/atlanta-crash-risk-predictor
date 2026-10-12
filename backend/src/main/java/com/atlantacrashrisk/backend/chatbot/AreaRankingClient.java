package com.atlantacrashrisk.backend.chatbot;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AreaRankingClient {
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private final String inferenceUrl;

    public AreaRankingClient(
        ObjectMapper objectMapper,
        @Value("${inference.url:http://127.0.0.1:8001}") String inferenceUrl
    ) {
        this.objectMapper = objectMapper;
        this.inferenceUrl = inferenceUrl;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(2))
            .build();
    }

    public AreaRankingResponse rank(String scenarioDatetime, Integer topK) {
        try {
            String body = objectMapper.writeValueAsString(new AreaRankingRequest(scenarioDatetime, topK));
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(inferenceUrl + "/areas/rank"))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                throw new IllegalStateException("Inference service returned HTTP " + response.statusCode());
            }
            return objectMapper.readValue(response.body(), AreaRankingResponse.class);
        } catch (IOException | InterruptedException | RuntimeException exception) {
            if (exception instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new IllegalStateException("Area ranking service is unavailable", exception);
        }
    }

    public record AreaRankingRequest(String scenarioDatetime, Integer topK) {}

    public record AreaRankingResponse(
        String scenarioDatetime,
        boolean conditionalOnRecordedCrash,
        AreaResult[] areas
    ) {}

    public record AreaResult(
        String id,
        double latitude,
        double longitude,
        double riskProbability
    ) {}
}