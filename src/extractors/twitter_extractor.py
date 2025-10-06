"""
Twitter extractor for social media sentiment analysis.
Extracts tweets related to Romanian financial markets and currency discussions.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

import tweepy
from tweepy import Client, Paginator
import aiohttp

from .base import BaseExtractor, ExtractorConfig, ExtractedContent


logger = logging.getLogger(__name__)


class TwitterExtractor(BaseExtractor):
    """Extractor for Twitter data related to financial markets."""

    def __init__(self, config: ExtractorConfig, db_session):
        super().__init__(config, db_session)
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not self.bearer_token:
            logger.warning("Twitter bearer token not found in environment variables")

        self.client = None
        if self.bearer_token:
            try:
                self.client = Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about Twitter data source."""
        return {
            "name": "Twitter Financial Sentiment Extractor",
            "type": "twitter",
            "keywords": [
                "#RON", "#EUR", "#USD", "#BTC", "#BNR",
                "leu romanesc", "curs valutar", "banca nationala",
                "economie romania", "piata valutara"
            ],
            "accounts": [
                "@BNROficial", "@MinisterulFinantelor", "@BNR",
                "@profit_ro", "@zf_ro", "@bursa_ro"
            ]
        }

    async def extract_tweets_by_keywords(self) -> List[ExtractedContent]:
        """Extract tweets using financial keywords."""
        if not self.client:
            logger.warning("Twitter client not available")
            return []

        contents = []
        financial_keywords = [
            "#RON", "#EUR", "#USD", "#BTC", "#BNR",
            "leu romanesc", "curs valutar", "banca nationala",
            "economie romania", "piata valutara", "investitii",
            "bursa bucuresti", "BET index"
        ]

        # Create search query
        query = " OR ".join(f'"{keyword}"' for keyword in financial_keywords)
        query += " -is:retweet lang:ro"  # Exclude retweets, Romanian language only

        try:
            # Get tweets from last 24 hours
            start_time = datetime.now() - timedelta(days=1)

            tweets = self.client.search_recent_tweets(
                query=query,
                start_time=start_time,
                max_results=min(100, self.config.batch_size),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                user_fields=['username', 'verified']
            )

            if tweets.data:
                for tweet in tweets.data:
                    # Get user info
                    user = None
                    if hasattr(tweet, 'author_id'):
                        try:
                            user = self.client.get_user(id=tweet.author_id).data
                        except:
                            pass

                    content = ExtractedContent(
                        source_id=f"twitter_{tweet.id}",
                        content=tweet.text,
                        title=f"Tweet by @{user.username if user else 'unknown'}",
                        url=f"https://twitter.com/i/web/status/{tweet.id}",
                        published_at=tweet.created_at,
                        metadata={
                            'platform': 'twitter',
                            'tweet_id': tweet.id,
                            'author_id': tweet.author_id,
                            'username': user.username if user else None,
                            'verified': user.verified if user else False,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'language': tweet.lang,
                            'extraction_method': 'keyword_search'
                        }
                    )
                    contents.append(content)

        except Exception as e:
            logger.error(f"Error extracting tweets by keywords: {str(e)}")

        return contents

    async def extract_tweets_by_accounts(self) -> List[ExtractedContent]:
        """Extract tweets from specific financial accounts."""
        if not self.client:
            logger.warning("Twitter client not available")
            return []

        contents = []
        financial_accounts = [
            "BNROficial", "MinisterulFinantelor", "BNR",
            "profit_ro", "zf_ro", "bursa_ro", "HotnewsRo",
            "Mediafax", "Agerpres"
        ]

        try:
            # Get tweets from last 12 hours for these accounts
            start_time = datetime.now() - timedelta(hours=12)

            for username in financial_accounts:
                try:
                    # Get user ID first
                    user_response = self.client.get_user(username=username)
                    if not user_response.data:
                        continue

                    user_id = user_response.data.id

                    # Get recent tweets
                    tweets = self.client.get_users_tweets(
                        id=user_id,
                        start_time=start_time,
                        max_results=min(20, self.config.batch_size // len(financial_accounts)),
                        tweet_fields=['created_at', 'public_metrics', 'lang'],
                        exclude=['retweets', 'replies']
                    )

                    if tweets.data:
                        for tweet in tweets.data:
                            content = ExtractedContent(
                                source_id=f"twitter_{tweet.id}",
                                content=tweet.text,
                                title=f"Tweet by @{username}",
                                url=f"https://twitter.com/{username}/status/{tweet.id}",
                                published_at=tweet.created_at,
                                metadata={
                                    'platform': 'twitter',
                                    'tweet_id': tweet.id,
                                    'username': username,
                                    'retweet_count': tweet.public_metrics['retweet_count'],
                                    'like_count': tweet.public_metrics['like_count'],
                                    'reply_count': tweet.public_metrics['reply_count'],
                                    'quote_count': tweet.public_metrics['quote_count'],
                                    'language': tweet.lang,
                                    'extraction_method': 'account_monitoring'
                                }
                            )
                            contents.append(content)

                except Exception as e:
                    logger.error(f"Error extracting tweets from @{username}: {str(e)}")

        except Exception as e:
            logger.error(f"Error extracting tweets by accounts: {str(e)}")

        return contents

    async def extract(self) -> List[ExtractedContent]:
        """Main extraction method."""
        all_contents = []

        # Extract tweets by keywords
        keyword_tweets = await self.extract_tweets_by_keywords()
        all_contents.extend(keyword_tweets)

        # Extract tweets from financial accounts
        account_tweets = await self.extract_tweets_by_accounts()
        all_contents.extend(account_tweets)

        # Remove duplicates based on tweet ID
        seen_ids = set()
        unique_contents = []
        for content in all_contents:
            tweet_id = content.metadata.get('tweet_id')
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                unique_contents.append(content)

        logger.info(f"Extracted {len(unique_contents)} unique tweets")
        return unique_contents

    def _is_financial_content(self, text: str) -> bool:
        """Check if tweet content is financial/economic related."""
        financial_indicators = [
            'ron', 'eur', 'usd', 'btc', 'leu', 'euro', 'dolar',
            'banca', 'burs', 'invest', 'economie', 'finant',
            'curs', 'valut', 'piat', 'actiun', 'obligatiun'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in financial_indicators)
