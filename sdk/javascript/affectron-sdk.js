/**
 * AffectRON JavaScript SDK
 * Provides easy integration with AffectRON API for web and Node.js applications
 */

class AffectRONSDK {
  constructor(apiKey, options = {}) {
    this.apiKey = apiKey;
    this.baseUrl = options.baseUrl || 'http://localhost:8000';
    this.timeout = options.timeout || 30000;

    // Headers for authenticated requests
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };

    // For Node.js environments
    if (typeof window === 'undefined') {
      this.fetch = require('node-fetch');
    }
  }

  /**
   * Analyze sentiment of text
   */
  async analyzeSentiment(text, language = 'ro') {
    const endpoint = `${this.baseUrl}/api/v1/sentiment/analyze`;
    const data = { text, language };

    try {
      const response = await this._makeRequest('POST', endpoint, data);

      return {
        text: response.text,
        sentiment: response.sentiment,
        entities: response.entities,
        language: response.language,
        marketRelevance: response.market_relevance,
        processedAt: response.processed_at
      };
    } catch (error) {
      throw new Error(`Sentiment analysis failed: ${error.message}`);
    }
  }

  /**
   * Get sentiment trends for currency
   */
  async getSentimentTrends(currency, timeframe = '24h') {
    const endpoint = `${this.baseUrl}/api/v1/sentiment`;
    const params = { currency, timeframe };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        currency: response.currency,
        sentiment: response.sentiment,
        confidence: response.confidence,
        trend: response.trend,
        timeframe: response.timeframe
      };
    } catch (error) {
      throw new Error(`Sentiment trends request failed: ${error.message}`);
    }
  }

  /**
   * Get current market rates
   */
  async getMarketRates(currencies) {
    const endpoint = `${this.baseUrl}/api/v1/market/rates`;
    const params = { currencies: currencies.join(',') };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        rates: response.rates,
        timestamp: response.timestamp,
        source: response.source
      };
    } catch (error) {
      throw new Error(`Market rates request failed: ${error.message}`);
    }
  }

  /**
   * Get market insights for currency
   */
  async getMarketInsights(currency, riskLevel = 'medium') {
    const endpoint = `${this.baseUrl}/api/v1/analytics/insights/market`;
    const params = { currency, risk_level: riskLevel };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        currency: response.currency,
        insights: response.insights,
        recommendations: response.recommendations,
        confidence: response.confidence,
        riskLevel: response.risk_level
      };
    } catch (error) {
      throw new Error(`Market insights request failed: ${error.message}`);
    }
  }

  /**
   * Get risk assessment for currency
   */
  async getRiskAssessment(currency) {
    const endpoint = `${this.baseUrl}/api/v1/analytics/risk`;
    const params = { currency };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        currency: response.currency,
        overallRiskScore: response.overall_risk_score,
        riskLevel: response.risk_level,
        riskFactors: response.risk_factors,
        recommendations: response.recommendations,
        confidence: response.confidence
      };
    } catch (error) {
      throw new Error(`Risk assessment request failed: ${error.message}`);
    }
  }

  /**
   * Get trend analysis
   */
  async getTrendAnalysis(timeframe = '24h') {
    const endpoint = `${this.baseUrl}/api/v1/analytics/trends`;
    const params = { timeframe };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        timeframe: response.timeframe,
        trends: response.trends,
        insights: response.insights,
        analysisDate: response.analysis_date
      };
    } catch (error) {
      throw new Error(`Trend analysis request failed: ${error.message}`);
    }
  }

  /**
   * Get active alerts
   */
  async getAlerts(limit = 50) {
    const endpoint = `${this.baseUrl}/api/v1/alerts/active`;
    const params = { limit };

    try {
      const response = await this._makeRequest('GET', endpoint, null, params);

      return {
        alerts: response.alerts.map(alert => ({
          id: alert.id,
          title: alert.title,
          message: alert.message,
          severity: alert.severity,
          type: alert.type,
          timestamp: alert.timestamp,
          acknowledged: alert.acknowledged
        }))
      };
    } catch (error) {
      throw new Error(`Alerts request failed: ${error.message}`);
    }
  }

  /**
   * Acknowledge an alert
   */
  async acknowledgeAlert(alertId) {
    const endpoint = `${this.baseUrl}/api/v1/alerts/${alertId}/acknowledge`;

    try {
      const response = await this._makeRequest('POST', endpoint);

      return {
        success: response.message === 'Alert acknowledged successfully'
      };
    } catch (error) {
      throw new Error(`Alert acknowledgment failed: ${error.message}`);
    }
  }

  /**
   * Get data sources status
   */
  async getDataSources() {
    const endpoint = `${this.baseUrl}/api/v1/data/sources`;

    try {
      const response = await this._makeRequest('GET', endpoint);

      return {
        sources: response.sources
      };
    } catch (error) {
      throw new Error(`Data sources request failed: ${error.message}`);
    }
  }

  /**
   * Get system status
   */
  async getSystemStatus() {
    const endpoint = `${this.baseUrl}/health`;

    try {
      const response = await this._makeRequest('GET', endpoint);

      return {
        status: response.status,
        timestamp: response.timestamp,
        version: response.version
      };
    } catch (error) {
      throw new Error(`System status request failed: ${error.message}`);
    }
  }

  /**
   * WebSocket connection for real-time data
   */
  connectWebSocket(subscriptions = [], onMessage, onError, onClose) {
    const wsUrl = this._getWebSocketUrl(subscriptions);

    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        console.log('Connected to AffectRON WebSocket');
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (onMessage) onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };

      socket.onclose = () => {
        console.log('WebSocket connection closed');
        if (onClose) onClose();
      };

      return socket;
    } catch (error) {
      throw new Error(`WebSocket connection failed: ${error.message}`);
    }
  }

  /**
   * Get WebSocket URL for subscriptions
   */
  _getWebSocketUrl(subscriptions) {
    const baseWsUrl = this.baseUrl.replace('http', 'ws');
    const params = subscriptions.map(sub => `subscribe=${sub}`).join('&');
    return params ? `${baseWsUrl}/ws?${params}` : `${baseWsUrl}/ws`;
  }

  /**
   * Make HTTP request
   */
  async _makeRequest(method, url, data = null, params = null) {
    let fetchUrl = url;

    // Add query parameters
    if (params) {
      const urlParams = new URLSearchParams(params);
      fetchUrl += `?${urlParams.toString()}`;
    }

    const options = {
      method,
      headers: this.headers,
      timeout: this.timeout
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(fetchUrl, options);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    return await response.json();
  }

  /**
   * Utility method to format sentiment result
   */
  formatSentimentResult(apiResponse) {
    return {
      text: apiResponse.text,
      sentiment: {
        label: apiResponse.sentiment.label,
        score: apiResponse.sentiment.score,
        confidence: apiResponse.sentiment.confidence,
        intensity: apiResponse.sentiment.intensity
      },
      entities: apiResponse.entities,
      language: apiResponse.language,
      marketRelevance: apiResponse.market_relevance,
      processedAt: apiResponse.processed_at
    };
  }

  /**
   * Utility method to format market data
   */
  formatMarketData(apiResponse) {
    return {
      rates: apiResponse.rates,
      timestamp: apiResponse.timestamp,
      source: apiResponse.source
    };
  }
}

// For Node.js environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AffectRONSDK;
}

// For browser environments
if (typeof window !== 'undefined') {
  window.AffectRONSDK = AffectRONSDK;
}
