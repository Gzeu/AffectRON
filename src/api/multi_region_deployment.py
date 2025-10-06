"""
Multi-region deployment system for AffectRON.
Provides global scalability, disaster recovery, and performance optimization.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp

from .global_currency import global_currency_manager
from .enhanced_websocket import connection_manager


@dataclass
class DeploymentRegion:
    """Deployment region configuration."""
    code: str  # 'us-east-1', 'eu-west-1', 'ap-southeast-1'
    name: str  # 'US East (N. Virginia)', 'EU West (Ireland)', etc.
    provider: str  # 'aws', 'gcp', 'azure'
    timezone: str
    primary_currencies: List[str]  # Currencies primarily served by this region

    # Infrastructure specs
    instance_type: str = 't3.large'
    min_instances: int = 1
    max_instances: int = 10

    # Performance targets
    target_latency_ms: int = 100
    target_availability: float = 99.9

    # Cost information
    cost_per_hour_usd: float = 0.10

    # Status
    is_active: bool = True
    current_instances: int = 0
    current_load: float = 0.0  # 0-1 scale


@dataclass
class GlobalLoadBalancer:
    """Global load balancer configuration."""
    algorithm: str  # 'round_robin', 'least_connections', 'geographic', 'latency_based'
    health_check_interval: int = 30  # seconds
    failover_enabled: bool = True
    regions_priority: List[str] = None  # Ordered list of region codes


class MultiRegionManager:
    """Manages multi-region deployment and global traffic routing."""

    def __init__(self):
        self.regions: Dict[str, DeploymentRegion] = {}
        self.load_balancer: GlobalLoadBalancer = None
        self.global_metrics: Dict[str, Any] = {}

        # Initialize regions
        self._initialize_regions()

        # Initialize load balancer
        self._initialize_load_balancer()

        self.logger = logging.getLogger(__name__)

    def _initialize_regions(self):
        """Initialize deployment regions."""
        regions_config = [
            DeploymentRegion(
                code='us-east-1',
                name='US East (N. Virginia)',
                provider='aws',
                timezone='America/New_York',
                primary_currencies=['USD', 'CAD', 'MXN'],
                instance_type='t3.large',
                min_instances=2,
                max_instances=20,
                target_latency_ms=50,
                cost_per_hour_usd=0.096
            ),
            DeploymentRegion(
                code='eu-west-1',
                name='EU West (Ireland)',
                provider='aws',
                timezone='Europe/Dublin',
                primary_currencies=['EUR', 'GBP', 'CHF', 'RON', 'PLN'],
                instance_type='t3.large',
                min_instances=3,
                max_instances=25,
                target_latency_ms=30,
                cost_per_hour_usd=0.096
            ),
            DeploymentRegion(
                code='ap-southeast-1',
                name='Asia Pacific (Singapore)',
                provider='aws',
                timezone='Asia/Singapore',
                primary_currencies=['JPY', 'AUD', 'NZD', 'SGD'],
                instance_type='t3.large',
                min_instances=2,
                max_instances=15,
                target_latency_ms=80,
                cost_per_hour_usd=0.112
            ),
            DeploymentRegion(
                code='us-west-2',
                name='US West (Oregon)',
                provider='aws',
                timezone='America/Los_Angeles',
                primary_currencies=['USD', 'CAD'],
                instance_type='t3.large',
                min_instances=1,
                max_instances=10,
                target_latency_ms=60,
                cost_per_hour_usd=0.096,
                is_active=False  # Backup region
            )
        ]

        for region in regions_config:
            self.regions[region.code] = region

    def _initialize_load_balancer(self):
        """Initialize global load balancer."""
        self.load_balancer = GlobalLoadBalancer(
            algorithm='geographic',
            health_check_interval=30,
            failover_enabled=True,
            regions_priority=['eu-west-1', 'us-east-1', 'ap-southeast-1', 'us-west-2']
        )

    async def route_request(self, request_data: Dict[str, Any]) -> str:
        """Route request to optimal region."""
        # Determine user location/preference
        user_currency = request_data.get('currency', 'EUR')
        user_location = await self._detect_user_location(request_data)

        # Find best region based on algorithm
        if self.load_balancer.algorithm == 'geographic':
            return self._route_geographic(user_currency, user_location)
        elif self.load_balancer.algorithm == 'latency_based':
            return await self._route_latency_based(user_location)
        elif self.load_balancer.algorithm == 'least_connections':
            return self._route_least_connections()
        else:
            return self._route_round_robin()

    def _route_geographic(self, currency: str, user_location: str) -> str:
        """Route based on geographic proximity and currency."""
        # Find regions that primarily serve this currency
        candidate_regions = []

        for region_code, region in self.regions.items():
            if region.is_active and currency in region.primary_currencies:
                candidate_regions.append(region_code)

        if candidate_regions:
            # Return first candidate (could be enhanced with load balancing)
            return candidate_regions[0]

        # Fallback to priority order
        for region_code in self.load_balancer.regions_priority:
            if region_code in self.regions and self.regions[region_code].is_active:
                return region_code

        return 'eu-west-1'  # Ultimate fallback

    async def _route_latency_based(self, user_location: str) -> str:
        """Route based on lowest latency."""
        # In production, this would ping all regions and measure latency
        # For now, use geographic routing as proxy
        return self._route_geographic('EUR', user_location)

    def _route_least_connections(self) -> str:
        """Route to region with least connections."""
        # Find region with lowest current load
        best_region = None
        lowest_load = float('inf')

        for region_code, region in self.regions.items():
            if region.is_active and region.current_load < lowest_load:
                lowest_load = region.current_load
                best_region = region_code

        return best_region or 'eu-west-1'

    def _route_round_robin(self) -> str:
        """Simple round-robin routing."""
        active_regions = [r for r in self.regions.values() if r.is_active]

        if not active_regions:
            return 'eu-west-1'

        # Simple round-robin (in production would track state)
        return active_regions[0].code

    async def _detect_user_location(self, request_data: Dict[str, Any]) -> str:
        """Detect user location from request."""
        # Check explicit location header
        location = request_data.get('headers', {}).get('x-user-location')

        if location:
            return location

        # Infer from currency or IP (simplified)
        currency = request_data.get('currency', 'EUR')

        if currency in ['USD', 'CAD', 'MXN']:
            return 'north_america'
        elif currency in ['JPY', 'AUD', 'NZD']:
            return 'asia_pacific'
        else:
            return 'europe'

    async def update_region_metrics(self, region_code: str, metrics: Dict[str, Any]):
        """Update metrics for region."""
        if region_code not in self.regions:
            return

        region = self.regions[region_code]

        # Update current instances and load
        region.current_instances = metrics.get('current_instances', region.current_instances)
        region.current_load = metrics.get('current_load', 0.0)

        # Store global metrics
        self.global_metrics[region_code] = {
            **metrics,
            'updated_at': datetime.now().isoformat()
        }

    async def perform_health_checks(self) -> Dict[str, bool]:
        """Perform health checks on all regions."""
        health_results = {}

        for region_code, region in self.regions.items():
            if not region.is_active:
                health_results[region_code] = False
                continue

            # In production, this would check actual endpoints
            # For now, simulate health check
            is_healthy = region.current_load < 0.9  # Healthy if not overloaded

            health_results[region_code] = is_healthy

            if not is_healthy:
                self.logger.warning(f"Region {region_code} health check failed")

        return health_results

    async def handle_failover(self, failed_region: str) -> str:
        """Handle failover when region fails."""
        if not self.load_balancer.failover_enabled:
            return failed_region

        self.logger.warning(f"Initiating failover from region: {failed_region}")

        # Mark region as inactive
        if failed_region in self.regions:
            self.regions[failed_region].is_active = False

        # Find alternative region
        alternative_region = None

        for region_code in self.load_balancer.regions_priority:
            if region_code != failed_region and region_code in self.regions:
                region = self.regions[region_code]
                if region.is_active and region.current_load < 0.8:  # Has capacity
                    alternative_region = region_code
                    break

        if alternative_region:
            self.logger.info(f"Failover to region: {alternative_region}")
            return alternative_region
        else:
            self.logger.error("No available regions for failover")
            return failed_region

    def get_deployment_costs(self) -> Dict[str, Any]:
        """Calculate deployment costs across regions."""
        total_monthly_cost = 0
        region_costs = {}

        for region_code, region in self.regions.items():
            if region.is_active:
                # Calculate based on current instances (assuming 24/7 operation)
                monthly_cost = region.current_instances * region.cost_per_hour_usd * 24 * 30
                region_costs[region_code] = {
                    'monthly_cost_usd': monthly_cost,
                    'current_instances': region.current_instances,
                    'instance_type': region.instance_type,
                    'cost_per_hour': region.cost_per_hour_usd
                }
                total_monthly_cost += monthly_cost

        return {
            'total_monthly_cost_usd': total_monthly_cost,
            'region_breakdown': region_costs,
            'cost_per_region': {
                region_code: costs['monthly_cost_usd']
                for region_code, costs in region_costs.items()
            },
            'calculated_at': datetime.now().isoformat()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get global performance metrics."""
        active_regions = [r for r in self.regions.values() if r.is_active]

        if not active_regions:
            return {'error': 'No active regions'}

        # Calculate averages
        avg_latency = sum(r.target_latency_ms for r in active_regions) / len(active_regions)
        avg_availability = sum(r.target_availability for r in active_regions) / len(active_regions)
        total_instances = sum(r.current_instances for r in active_regions)
        avg_load = sum(r.current_load for r in active_regions) / len(active_regions)

        return {
            'active_regions': len(active_regions),
            'total_instances': total_instances,
            'average_latency_ms': avg_latency,
            'average_availability_percent': avg_availability,
            'average_load_percent': avg_load * 100,
            'regions': {
                region_code: {
                    'current_instances': region.current_instances,
                    'current_load': region.current_load,
                    'target_latency_ms': region.target_latency_ms,
                    'is_healthy': region.current_load < 0.9
                }
                for region_code, region in self.regions.items()
            },
            'metrics_timestamp': datetime.now().isoformat()
        }

    def optimize_deployment(self) -> Dict[str, Any]:
        """Optimize deployment across regions."""
        recommendations = []

        for region_code, region in self.regions.items():
            if not region.is_active:
                continue

            # Check if region is overloaded
            if region.current_load > 0.8:
                recommended_instances = min(region.current_instances + 1, region.max_instances)
                if recommended_instances > region.current_instances:
                    recommendations.append({
                        'region': region_code,
                        'action': 'scale_up',
                        'current_instances': region.current_instances,
                        'recommended_instances': recommended_instances,
                        'reason': 'High load detected'
                    })

            # Check if region is underutilized
            elif region.current_load < 0.3 and region.current_instances > region.min_instances:
                recommended_instances = max(region.current_instances - 1, region.min_instances)
                recommendations.append({
                    'region': region_code,
                    'action': 'scale_down',
                    'current_instances': region.current_instances,
                    'recommended_instances': recommended_instances,
                    'reason': 'Low utilization'
                })

        return {
            'optimization_recommendations': recommendations,
            'potential_cost_savings': self._calculate_potential_savings(recommendations),
            'optimization_timestamp': datetime.now().isoformat()
        }

    def _calculate_potential_savings(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate potential cost savings from optimization."""
        savings = 0.0

        for rec in recommendations:
            if rec['action'] == 'scale_down':
                instances_reduced = rec['current_instances'] - rec['recommended_instances']
                region = self.regions[rec['region']]
                savings += instances_reduced * region.cost_per_hour_usd * 24 * 30  # Monthly

        return savings

    async def sync_global_data(self):
        """Sync data across all regions."""
        # This would sync:
        # - Currency exchange rates
        # - Sentiment data
        # - User preferences
        # - System configurations

        sync_data = {
            'exchange_rates': global_currency_manager.exchange_rates,
            'global_metrics': self.global_metrics,
            'sync_timestamp': datetime.now().isoformat()
        }

        # In production, this would broadcast to all regions
        # For now, just log the sync
        self.logger.info(f"Global data sync initiated: {len(sync_data)} data types")

        return sync_data

    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status."""
        return {
            'regions': {
                region_code: {
                    'name': region.name,
                    'provider': region.provider,
                    'is_active': region.is_active,
                    'current_instances': region.current_instances,
                    'current_load': region.current_load,
                    'primary_currencies': region.primary_currencies,
                    'target_latency_ms': region.target_latency_ms
                }
                for region_code, region in self.regions.items()
            },
            'load_balancer': {
                'algorithm': self.load_balancer.algorithm,
                'failover_enabled': self.load_balancer.failover_enabled,
                'regions_priority': self.load_balancer.regions_priority
            },
            'global_metrics': self.global_metrics,
            'deployment_costs': self.get_deployment_costs(),
            'performance_metrics': self.get_performance_metrics(),
            'last_updated': datetime.now().isoformat()
        }


# Global multi-region manager instance
multi_region_manager = MultiRegionManager()


async def initialize_multi_region_deployment():
    """Initialize multi-region deployment system."""
    # Perform initial health checks
    health_results = await multi_region_manager.perform_health_checks()

    unhealthy_regions = [region for region, healthy in health_results.items() if not healthy]

    if unhealthy_regions:
        logging.getLogger(__name__).warning(f"Unhealthy regions detected: {unhealthy_regions}")

    logging.getLogger(__name__).info("Multi-region deployment system initialized")


def get_multi_region_status():
    """Get multi-region deployment status."""
    return {
        'initialized': True,
        'regions_count': len(multi_region_manager.regions),
        'active_regions': len([r for r in multi_region_manager.regions.values() if r.is_active]),
        'global_status': multi_region_manager.get_global_status()
    }
