"""
Risk scoring analytics for AffectRON.
Calculates risk scores based on sentiment volatility, market conditions, and other factors.
"""

import json

from .base import BaseAnalytics, AnalyticsConfig, AnalyticsResult


logger = logging.getLogger(__name__)


class RiskScoringAnalytics(BaseAnalytics):
    """Analytics module for calculating financial risk scores."""

    def __init__(self, config: AnalyticsConfig, db_session):
        super().__init__(config, db_session)

        # Risk scoring parameters
        self.risk_weights = {
            'sentiment_volatility': 0.3,
            'market_volatility': 0.25,
            'volume_anomaly': 0.2,
            'entity_diversity': 0.15,
            'time_consistency': 0.1
        }

    async def analyze(self) -> List[AnalyticsResult]:
        """Calculate risk scores."""
        analysis_data = self.get_analysis_data()

        if analysis_data['record_count'] == 0:
            return []

        # Calculate overall risk score
        risk_components = self._calculate_risk_components(analysis_data)

        # Generate risk assessment
        overall_risk = self._calculate_overall_risk(risk_components)

        # Create risk alerts if needed
        self._generate_risk_alerts(overall_risk, risk_components)

        insights = {
            'overall_risk_score': overall_risk,
            'risk_components': risk_components,
            'risk_level': self._categorize_risk_level(overall_risk),
            'risk_trends': self._analyze_risk_trends(analysis_data),
            'recommendations': self._generate_risk_recommendations(overall_risk, risk_components)
        }

        return [AnalyticsResult(
            analytics_name=self.config.name,
            result_type="risk_assessment",
            insights=insights,
            confidence=min(0.9, analysis_data['record_count'] / 100)  # Confidence based on data volume
        )]

    def _calculate_risk_components(self, analysis_data) -> Dict[str, float]:
        """Calculate individual risk components."""
        components = {}

        # Sentiment volatility risk
        sentiment_scores = [sentiment.sentiment_score for _, sentiment in analysis_data['sentiment_data']]
        if sentiment_scores:
            sentiment_volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            components['sentiment_volatility'] = min(sentiment_volatility * 2, 1.0)  # Scale to 0-1
        else:
            components['sentiment_volatility'] = 0.0

        # Market volatility risk
        if analysis_data['market_data']:
            market_rates = [data.rate for data in analysis_data['market_data']]
            if len(market_rates) > 1:
                market_volatility = statistics.stdev(market_rates) / statistics.mean(market_rates) if statistics.mean(market_rates) > 0 else 0
                components['market_volatility'] = min(market_volatility * 5, 1.0)  # Scale to 0-1
            else:
                components['market_volatility'] = 0.0
        else:
            components['market_volatility'] = 0.5  # Medium risk when no market data

        # Volume anomaly risk
        components['volume_anomaly'] = self._calculate_volume_anomaly_risk(analysis_data)

        # Entity diversity risk
        components['entity_diversity'] = self._calculate_entity_diversity_risk(analysis_data)

        # Time consistency risk
        components['time_consistency'] = self._calculate_time_consistency_risk(analysis_data)

        return components

    def _calculate_volume_anomaly_risk(self, analysis_data) -> float:
        """Calculate risk based on data volume anomalies."""
        if not analysis_data['sentiment_data']:
            return 0.5  # Medium risk for no data

        # Group by hour and calculate volumes
        hourly_volumes = defaultdict(int)
        for _, sentiment in analysis_data['sentiment_data']:
            hour_key = sentiment.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_volumes[hour_key] += 1

        volumes = list(hourly_volumes.values())

        if len(volumes) < 2:
            return 0.1  # Low risk for insufficient data

        mean_volume = statistics.mean(volumes)
        std_volume = statistics.stdev(volumes)

        if std_volume == 0:
            return 0.1  # Low risk for consistent volume

        # Calculate coefficient of variation
        cv = std_volume / mean_volume

        # Risk increases with volume variability
        return min(cv * 2, 1.0)

    def _calculate_entity_diversity_risk(self, analysis_data) -> float:
        """Calculate risk based on entity diversity."""
        entity_counts = defaultdict(int)

        for data, sentiment in analysis_data['sentiment_data']:
            if sentiment.entities:
                try:
                    entities = json.loads(sentiment.entities) if isinstance(sentiment.entities, str) else sentiment.entities

                    for entity_list in entities.values():
                        if isinstance(entity_list, list):
                            for entity in entity_list:
                                if isinstance(entity, dict) and 'text' in entity:
                                    entity_counts[entity['text']] += 1

                except (json.JSONDecodeError, KeyError):
                    continue

        if not entity_counts:
            return 0.3  # Medium risk for no entity data

        # Calculate diversity (higher diversity = lower risk)
        total_entities = sum(entity_counts.values())
        unique_entities = len(entity_counts)

        if unique_entities == 0:
            return 0.5

        # Normalize diversity score (0-1, where 1 is maximum diversity)
        diversity_score = min(unique_entities / max(total_entities * 0.1, 1), 1.0)

        # Risk is inverse of diversity (less diversity = higher risk)
        return 1.0 - diversity_score

    def _calculate_time_consistency_risk(self, analysis_data) -> float:
        """Calculate risk based on temporal consistency."""
        if not analysis_data['sentiment_data']:
            return 0.5

        timestamps = [sentiment.created_at for _, sentiment in analysis_data['sentiment_data']]

        if len(timestamps) < 2:
            return 0.1

        sorted_timestamps = sorted(timestamps)

        # Calculate gaps in data collection
        gaps = []
        for i in range(1, len(sorted_timestamps)):
            gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            gaps.append(gap)

        if not gaps:
            return 0.0

        mean_gap = statistics.mean(gaps)
        max_gap = max(gaps)

        # Risk based on data collection gaps
        # Larger gaps = higher risk (less consistent monitoring)
        gap_risk = min(max_gap / 3600, 1.0)  # Normalize to hours

        return gap_risk

    def _calculate_overall_risk(self, risk_components: Dict[str, float]) -> float:
        """Calculate overall risk score."""
        if not risk_components:
            return 0.5  # Medium risk for no data

        # Weighted average of risk components
        total_weight = sum(self.risk_weights.values())
        weighted_risk = sum(
            risk_components.get(component, 0) * weight
            for component, weight in self.risk_weights.items()
        )

        overall_risk = weighted_risk / total_weight

        # Ensure risk is between 0 and 1
        return max(0.0, min(1.0, overall_risk))

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on score."""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"

    def _analyze_risk_trends(self, analysis_data) -> Dict[str, Any]:
        """Analyze risk trends over time."""
        # This would implement time-series analysis of risk components
        # For now, return basic trend information
        return {
            'trend_analysis': 'not_implemented',
            'trend_direction': 'stable',
            'confidence': 0.7
        }

    def _generate_risk_recommendations(self, overall_risk: float, risk_components: Dict[str, float]) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []

        if overall_risk > 0.7:
            recommendations.append("Consider reducing position sizes due to high overall risk")
            recommendations.append("Increase monitoring frequency")

        if risk_components.get('sentiment_volatility', 0) > 0.6:
            recommendations.append("High sentiment volatility detected - consider hedging strategies")

        if risk_components.get('market_volatility', 0) > 0.6:
            recommendations.append("Market volatility is elevated - review stop-loss orders")

        if risk_components.get('volume_anomaly', 0) > 0.5:
            recommendations.append("Unusual data volume patterns - verify data sources")

        if risk_components.get('entity_diversity', 0) > 0.6:
            recommendations.append("Low entity diversity - consider expanding information sources")

        if risk_components.get('time_consistency', 0) > 0.5:
            recommendations.append("Inconsistent data collection - review monitoring schedule")

        if not recommendations:
            recommendations.append("Risk levels are within normal parameters - maintain current strategy")

        return recommendations

    def _generate_risk_alerts(self, overall_risk: float, risk_components: Dict[str, float]):
        """Generate alerts based on risk levels."""
        if overall_risk > 0.8:
            self.create_alert(
                alert_type="risk",
                severity="critical",
                title="Critical Risk Level Detected",
                message=f"Overall risk score of {overall_risk".2f"} exceeds critical threshold",
                data={'risk_score': overall_risk, 'components': risk_components}
            )

        elif overall_risk > 0.6:
            self.create_alert(
                alert_type="risk",
                severity="high",
                title="High Risk Level Detected",
                message=f"Overall risk score of {overall_risk".2f"} indicates elevated risk",
                data={'risk_score': overall_risk, 'components': risk_components}
            )

        # Component-specific alerts
        for component, score in risk_components.items():
            if score > 0.7:
                self.create_alert(
                    alert_type="risk",
                    severity="medium",
                    title=f"High {component.replace('_', ' ').title()} Risk",
                    message=f"{component.replace('_', ' ').title()} risk score of {score".2f"} is elevated",
                    data={'component': component, 'score': score}
                )
