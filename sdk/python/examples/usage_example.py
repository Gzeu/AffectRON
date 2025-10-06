#!/usr/bin/env python3
"""
Example usage of AffectRON Python SDK.
Demonstrates various SDK features for sentiment analysis and market data.
"""

import asyncio
import os
from affectron_sdk import AffectRONSDK, quick_sentiment_analysis, quick_market_rates


async def main():
    """Main example function."""

    # Initialize SDK
    api_key = os.getenv('AFFECTRON_API_KEY', 'your-api-key-here')
    sdk = AffectRONSDK(api_key, base_url="http://localhost:8000")

    print("🤖 AffectRON Python SDK Examples")
    print("=" * 50)

    # Example 1: Sentiment Analysis
    print("\n1. 📝 Sentiment Analysis Example")
    print("-" * 30)

    text = "RON exchange rate increased significantly today after BNR announcement"
    print(f"Analyzing text: '{text}'")

    async with sdk:
        sentiment_result = await sdk.analyze_sentiment(text, language="ro")

    print(f"Sentiment: {sentiment_result['sentiment']['label']}")
    print(f"Score: {sentiment_result['sentiment']['score']".3f"}")
    print(f"Confidence: {sentiment_result['sentiment']['confidence']".3f"}")
    print(f"Market Relevance: {sentiment_result['market_relevance']".3f"}")

    # Example 2: Market Rates
    print("\n2. 💱 Market Rates Example")
    print("-" * 30)

    currencies = ["RON", "EUR", "USD"]

    async with sdk:
        market_result = await sdk.get_market_rates(currencies)

    print("Current exchange rates:")
    for pair, data in market_result['rates'].items():
        print(f"  {pair}: {data['rate']".4f"} (Change: {data['change_percent']"+.2f"}%)")

    # Example 3: Market Insights
    print("\n3. 📊 Market Insights Example")
    print("-" * 30)

    async with sdk:
        insights_result = await sdk.get_market_insights("RON", risk_level="medium")

    print(f"Currency: {insights_result['currency']}")
    print(f"Risk Level: {insights_result['risk_level']}")
    print(f"Confidence: {insights_result['confidence']".3f"}")
    print("Insights:")
    for insight in insights_result['insights'][:3]:  # Show first 3
        print(f"  • {insight}")

    # Example 4: Risk Assessment
    print("\n4. ⚠️  Risk Assessment Example")
    print("-" * 30)

    async with sdk:
        risk_result = await sdk.get_risk_assessment("RON")

    print(f"Currency: {risk_result['currency']}")
    print(f"Overall Risk Score: {risk_result['overall_risk_score']".3f"}")
    print(f"Risk Level: {risk_result['risk_level']}")
    print("Risk Factors:")
    for factor, score in list(risk_result['risk_factors'].items())[:5]:  # Show first 5
        print(f"  {factor}: {score".3f"}")

    # Example 5: Real-time Alerts
    print("\n5. 🚨 Real-time Alerts Example")
    print("-" * 30)

    async with sdk:
        alerts_result = await sdk.get_alerts(limit=5)

    print(f"Active alerts: {len(alerts_result['alerts'])}")
    for alert in alerts_result['alerts'][:3]:  # Show first 3
        print(f"  {alert['severity'].upper()}: {alert['title']}")

    # Example 6: WebSocket Real-time Streaming
    print("\n6. 📡 WebSocket Real-time Streaming")
    print("-" * 35)

    async def message_handler(data):
        """Handle real-time WebSocket messages."""
        if data.get('type') == 'sentiment_update':
            print(f"  📈 Sentiment Update: {data['data'].get('currency', 'Unknown')}")
        elif data.get('type') == 'market_data':
            market_info = data['data']
            print(f"  💱 Market Update: {market_info.get('currency_pair', 'Unknown')} = {market_info.get('rate', 0)".4f"}")
        elif data.get('type') == 'alert':
            alert_info = data['data']
            print(f"  🚨 Alert: {alert_info.get('title', 'Unknown')}")

    print("Starting WebSocket stream (press Ctrl+C to stop)...")

    try:
        # Note: WebSocket streaming would require websockets package
        # For this example, we'll just show the URL
        ws_url = sdk.get_websocket_url(['sentiment', 'market_rates'])
        print(f"WebSocket URL: {ws_url}")
        print("In production, this would connect and stream real-time data")

        # Simulate some streaming for demo
        await asyncio.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping WebSocket stream...")

    # Example 7: Quick Analysis Functions
    print("\n7. ⚡ Quick Analysis Functions")
    print("-" * 30)

    # Quick sentiment analysis (no SDK initialization needed)
    quick_result = await quick_sentiment_analysis(api_key, "RON shows strong bullish signals today")
    print(f"Quick sentiment: {quick_result.sentiment_label} ({quick_result.sentiment_score".3f"})")

    # Quick market rates
    quick_rates = await quick_market_rates(api_key, ["EUR", "USD"])
    print("Quick market rates:")
    for rate_data in quick_rates:
        print(f"  {rate_data.currency_pair}: {rate_data.rate".4f"}")

    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
