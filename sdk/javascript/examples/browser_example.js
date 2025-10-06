/**
 * AffectRON JavaScript SDK Usage Examples
 * Demonstrates various SDK features for web applications
 */

class AffectRONExample {
  constructor() {
    // Initialize SDK with your API key
    this.sdk = new AffectRONSDK('your-api-key-here', {
      baseUrl: 'http://localhost:8000',
      timeout: 30000
    });
  }

  /**
   * Example 1: Sentiment Analysis
   */
  async exampleSentimentAnalysis() {
    console.log('📝 Sentiment Analysis Example');

    const text = 'RON exchange rate increased significantly after BNR announcement';
    const language = 'ro';

    try {
      const result = await this.sdk.analyzeSentiment(text, language);

      console.log('Text:', result.text);
      console.log('Sentiment:', result.sentiment.label);
      console.log('Score:', result.sentiment.score.toFixed(3));
      console.log('Confidence:', result.sentiment.confidence.toFixed(3));
      console.log('Market Relevance:', result.marketRelevance.toFixed(3));

      return result;
    } catch (error) {
      console.error('Sentiment analysis failed:', error.message);
    }
  }

  /**
   * Example 2: Market Rates
   */
  async exampleMarketRates() {
    console.log('💱 Market Rates Example');

    const currencies = ['RON', 'EUR', 'USD'];

    try {
      const result = await this.sdk.getMarketRates(currencies);

      console.log('Market rates:');
      Object.entries(result.rates).forEach(([pair, data]) => {
        console.log(`${pair}: ${data.rate.toFixed(4)} (Change: ${data.change_percent.toFixed(2)}%)`);
      });

      return result;
    } catch (error) {
      console.error('Market rates request failed:', error.message);
    }
  }

  /**
   * Example 3: Market Insights
   */
  async exampleMarketInsights() {
    console.log('📊 Market Insights Example');

    const currency = 'RON';
    const riskLevel = 'medium';

    try {
      const result = await this.sdk.getMarketInsights(currency, riskLevel);

      console.log(`Currency: ${result.currency}`);
      console.log(`Risk Level: ${result.riskLevel}`);
      console.log(`Confidence: ${result.confidence.toFixed(3)}`);
      console.log('Insights:');
      result.insights.forEach(insight => {
        console.log(`  • ${insight}`);
      });

      return result;
    } catch (error) {
      console.error('Market insights request failed:', error.message);
    }
  }

  /**
   * Example 4: Risk Assessment
   */
  async exampleRiskAssessment() {
    console.log('⚠️ Risk Assessment Example');

    const currency = 'RON';

    try {
      const result = await this.sdk.getRiskAssessment(currency);

      console.log(`Currency: ${result.currency}`);
      console.log(`Overall Risk Score: ${result.overallRiskScore.toFixed(3)}`);
      console.log(`Risk Level: ${result.riskLevel}`);
      console.log('Risk Factors:');
      Object.entries(result.riskFactors).forEach(([factor, score]) => {
        console.log(`  ${factor}: ${score.toFixed(3)}`);
      });

      return result;
    } catch (error) {
      console.error('Risk assessment failed:', error.message);
    }
  }

  /**
   * Example 5: Real-time Alerts
   */
  async exampleAlerts() {
    console.log('🚨 Alerts Example');

    try {
      const result = await this.sdk.getAlerts(10);

      console.log(`Active alerts: ${result.alerts.length}`);
      result.alerts.forEach(alert => {
        console.log(`${alert.severity.toUpperCase()}: ${alert.title}`);
      });

      return result;
    } catch (error) {
      console.error('Alerts request failed:', error.message);
    }
  }

  /**
   * Example 6: WebSocket Real-time Streaming
   */
  exampleWebSocketStreaming() {
    console.log('📡 WebSocket Streaming Example');

    const subscriptions = ['sentiment', 'market_rates', 'alerts'];

    const socket = this.sdk.connectWebSocket(
      subscriptions,
      (data) => {
        // Handle incoming real-time data
        console.log('Received:', data.type, data.data);
      },
      (error) => {
        console.error('WebSocket error:', error);
      },
      () => {
        console.log('WebSocket connection closed');
      }
    );

    // Return socket for manual management if needed
    return socket;
  }

  /**
   * Example 7: Batch Operations
   */
  async exampleBatchOperations() {
    console.log('🔄 Batch Operations Example');

    const currencies = ['RON', 'EUR', 'USD'];

    try {
      // Get sentiment for multiple currencies
      const sentimentPromises = currencies.map(currency =>
        this.sdk.getSentimentTrends(currency, '24h')
      );

      const sentimentResults = await Promise.all(sentimentPromises);

      console.log('Batch sentiment results:');
      sentimentResults.forEach((result, index) => {
        console.log(`${currencies[index]}: ${result.sentiment.label} (${result.sentiment.score.toFixed(3)})`);
      });

      return sentimentResults;
    } catch (error) {
      console.error('Batch operations failed:', error.message);
    }
  }

  /**
   * Example 8: Error Handling
   */
  async exampleErrorHandling() {
    console.log('❌ Error Handling Example');

    try {
      // Try to analyze empty text (should fail)
      await this.sdk.analyzeSentiment('');
    } catch (error) {
      console.log('Caught expected error:', error.message);
    }

    try {
      // Try to get rates for invalid currency
      await this.sdk.getMarketRates(['INVALID']);
    } catch (error) {
      console.log('Caught validation error:', error.message);
    }
  }

  /**
   * Run all examples
   */
  async runAllExamples() {
    console.log('🤖 AffectRON JavaScript SDK - All Examples');
    console.log('=' * 50);

    await this.exampleSentimentAnalysis();
    await this.exampleMarketRates();
    await this.exampleMarketInsights();
    await this.exampleRiskAssessment();
    await this.exampleAlerts();
    await this.exampleBatchOperations();
    await this.exampleErrorHandling();

    // WebSocket example (doesn't need await)
    this.exampleWebSocketStreaming();

    console.log('✅ All examples completed!');
  }
}

// Browser usage example
if (typeof window !== 'undefined') {
  // Make example available globally
  window.AffectRONExample = AffectRONExample;

  // Auto-run example when page loads (for demo purposes)
  window.addEventListener('load', async () => {
    const example = new AffectRONExample();
    await example.runAllExamples();
  });
}

// Node.js usage example
if (typeof module !== 'undefined' && module.exports) {
  const example = new AffectRONExample();
  example.runAllExamples().catch(console.error);
}
