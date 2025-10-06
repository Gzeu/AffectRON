package com.affectron.sdk;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * AffectRON Java SDK
 * Provides easy integration with AffectRON API for Java applications
 */
public class AffectRONSDK {

    private final String apiKey;
    private final String baseUrl;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public AffectRONSDK(String apiKey) {
        this(apiKey, "http://localhost:8000");
    }

    public AffectRONSDK(String apiKey, String baseUrl) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;

        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
    }

    // Data classes for API responses
    public static class SentimentResult {
        public String text;
        public SentimentData sentiment;
        public Map<String, List<String>> entities;
        public String language;
        public double market_relevance;
        public String processed_at;

        public static class SentimentData {
            public String label;
            public double score;
            public double confidence;
            public String intensity;
        }
    }

    public static class MarketRatesResult {
        public Map<String, RateData> rates;
        public String timestamp;
        public String source;

        public static class RateData {
            public double rate;
            public double change;
            public double change_percent;
            public Long volume;
        }
    }

    public static class MarketInsightResult {
        public String currency;
        public List<String> insights;
        public List<String> recommendations;
        public double confidence;
        public String risk_level;
    }

    public static class RiskAssessmentResult {
        public String currency;
        public double overall_risk_score;
        public String risk_level;
        public Map<String, Double> risk_factors;
        public List<String> recommendations;
        public double confidence;
    }

    public static class AlertResult {
        public String id;
        public String title;
        public String message;
        public String severity;
        public String type;
        public String timestamp;
        public boolean acknowledged;
    }

    /**
     * Analyze sentiment of text
     */
    public CompletableFuture<SentimentResult> analyzeSentiment(String text, String language) {
        return makeRequest("POST", "/api/v1/sentiment/analyze",
                Map.of("text", text, "language", language))
                .thenApply(this::parseResponse);
    }

    /**
     * Get sentiment trends for currency
     */
    public CompletableFuture<SentimentResult> getSentimentTrends(String currency, String timeframe) {
        return makeRequest("GET", "/api/v1/sentiment",
                Map.of("currency", currency, "timeframe", timeframe))
                .thenApply(this::parseResponse);
    }

    /**
     * Get current market rates
     */
    public CompletableFuture<MarketRatesResult> getMarketRates(List<String> currencies) {
        String currenciesParam = String.join(",", currencies);
        return makeRequest("GET", "/api/v1/market/rates",
                Map.of("currencies", currenciesParam))
                .thenApply(this::parseResponse);
    }

    /**
     * Get market insights for currency
     */
    public CompletableFuture<MarketInsightResult> getMarketInsights(String currency, String riskLevel) {
        return makeRequest("GET", "/api/v1/analytics/insights/market",
                Map.of("currency", currency, "risk_level", riskLevel))
                .thenApply(this::parseResponse);
    }

    /**
     * Get risk assessment for currency
     */
    public CompletableFuture<RiskAssessmentResult> getRiskAssessment(String currency) {
        return makeRequest("GET", "/api/v1/analytics/risk",
                Map.of("currency", currency))
                .thenApply(this::parseResponse);
    }

    /**
     * Get trend analysis
     */
    public CompletableFuture<Map<String, Object>> getTrendAnalysis(String timeframe) {
        return makeRequest("GET", "/api/v1/analytics/trends",
                Map.of("timeframe", timeframe))
                .thenApply(this::parseResponse);
    }

    /**
     * Get active alerts
     */
    public CompletableFuture<List<AlertResult>> getAlerts(int limit) {
        return makeRequest("GET", "/api/v1/alerts/active",
                Map.of("limit", String.valueOf(limit)))
                .thenApply(response -> {
                    try {
                        Map<String, Object> responseMap = objectMapper.readValue(response, Map.class);
                        List<Map<String, Object>> alerts = (List<Map<String, Object>>) responseMap.get("alerts");
                        return alerts.stream()
                                .map(this::mapToAlertResult)
                                .toList();
                    } catch (Exception e) {
                        throw new RuntimeException("Failed to parse alerts response", e);
                    }
                });
    }

    /**
     * Acknowledge an alert
     */
    public CompletableFuture<Boolean> acknowledgeAlert(String alertId) {
        return makeRequest("POST", "/api/v1/alerts/" + alertId + "/acknowledge", Map.of())
                .thenApply(response -> {
                    try {
                        Map<String, Object> responseMap = objectMapper.readValue(response, Map.class);
                        return "Alert acknowledged successfully".equals(responseMap.get("message"));
                    } catch (Exception e) {
                        return false;
                    }
                });
    }

    /**
     * Get data sources status
     */
    public CompletableFuture<Map<String, Object>> getDataSources() {
        return makeRequest("GET", "/api/v1/data/sources", Map.of())
                .thenApply(this::parseResponse);
    }

    /**
     * Get system status
     */
    public CompletableFuture<Map<String, Object>> getSystemStatus() {
        return makeRequest("GET", "/health", Map.of())
                .thenApply(this::parseResponse);
    }

    /**
     * Make HTTP request
     */
    private CompletableFuture<String> makeRequest(String method, String endpoint, Map<String, String> params) {
        StringBuilder urlBuilder = new StringBuilder(baseUrl + endpoint);

        if (!params.isEmpty()) {
            urlBuilder.append("?");
            params.forEach((key, value) ->
                urlBuilder.append(key).append("=").append(value).append("&"));
            urlBuilder.setLength(urlBuilder.length() - 1); // Remove last &
        }

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(urlBuilder.toString()))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .timeout(Duration.ofSeconds(30));

        HttpRequest request;

        if ("POST".equals(method) && !params.isEmpty()) {
            String jsonBody = params.entrySet().stream()
                    .map(e -> "\"" + e.getKey() + "\":\"" + e.getValue() + "\"")
                    .reduce((a, b) -> a + "," + b)
                    .orElse("");
            jsonBody = "{" + jsonBody + "}";

            request = requestBuilder
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
        } else {
            request = requestBuilder.GET().build();
        }

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() >= 200 && response.statusCode() < 300) {
                        return response.body();
                    } else {
                        throw new RuntimeException("HTTP " + response.statusCode() + ": " + response.body());
                    }
                });
    }

    /**
     * Parse JSON response
     */
    private <T> T parseResponse(String responseBody) {
        try {
            return objectMapper.readValue(responseBody, new com.fasterxml.jackson.core.type.TypeReference<T>() {});
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse response: " + responseBody, e);
        }
    }

    /**
     * Map alert response to AlertResult
     */
    private AlertResult mapToAlertResult(Map<String, Object> alertMap) {
        AlertResult result = new AlertResult();
        result.id = (String) alertMap.get("id");
        result.title = (String) alertMap.get("title");
        result.message = (String) alertMap.get("message");
        result.severity = (String) alertMap.get("severity");
        result.type = (String) alertMap.get("type");
        result.timestamp = (String) alertMap.get("timestamp");
        result.acknowledged = (Boolean) alertMap.get("acknowledged");
        return result;
    }

    /**
     * Get WebSocket URL for real-time subscriptions
     */
    public String getWebSocketUrl(List<String> subscriptions) {
        String wsUrl = baseUrl.replace("http", "ws");
        if (subscriptions != null && !subscriptions.isEmpty()) {
            String params = subscriptions.stream()
                    .map(sub -> "subscribe=" + sub)
                    .reduce((a, b) -> a + "&" + b)
                    .orElse("");
            return wsUrl + "/ws?" + params;
        }
        return wsUrl + "/ws";
    }

    /**
     * Synchronous version of analyzeSentiment for compatibility
     */
    public SentimentResult analyzeSentimentSync(String text, String language) {
        try {
            return analyzeSentiment(text, language).get();
        } catch (Exception e) {
            throw new RuntimeException("Sentiment analysis failed", e);
        }
    }

    /**
     * Synchronous version of getMarketRates for compatibility
     */
    public MarketRatesResult getMarketRatesSync(List<String> currencies) {
        try {
            return getMarketRates(currencies).get();
        } catch (Exception e) {
            throw new RuntimeException("Market rates request failed", e);
        }
    }
}
