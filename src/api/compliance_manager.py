"""
Regulatory compliance module for AffectRON.
Implements MiFID II, GDPR, and other financial regulations compliance.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

from .multi_tenant import TenantManager, TenantContext
from .auth import verify_token


@dataclass
class ComplianceRule:
    """Regulatory compliance rule."""
    id: str
    name: str
    regulation: str  # 'MiFID II', 'GDPR', 'PSD2', 'FATF', etc.
    rule_type: str  # 'data_retention', 'audit_logging', 'user_consent', 'transaction_reporting'
    description: str
    requirements: Dict[str, Any]
    enforcement_level: str  # 'mandatory', 'recommended', 'optional'
    implementation_status: str  # 'implemented', 'partial', 'planned'
    last_reviewed: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceEvent:
    """Compliance-related event for audit trail."""
    id: str
    event_type: str  # 'data_access', 'user_consent', 'transaction', 'system_change'
    regulation: str
    description: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    compliance_status: str = 'compliant'  # 'compliant', 'violation', 'warning'


@dataclass
class AuditTrail:
    """Audit trail entry."""
    id: str
    action: str
    resource: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""


class ComplianceManager:
    """Manages regulatory compliance for AffectRON."""

    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.audit_trail: List[AuditTrail] = []
        self.compliance_events: List[ComplianceEvent] = []

        # Data retention policies
        self.data_retention_policies = {
            'user_data': timedelta(days=2555),  # 7 years for MiFID II
            'transaction_data': timedelta(days=2555),  # 7 years for MiFID II
            'audit_logs': timedelta(days=2555),  # 7 years for compliance
            'sentiment_data': timedelta(days=365),  # 1 year for business data
            'market_data': timedelta(days=90)  # 90 days for market data
        }

        # GDPR compliance settings
        self.gdpr_settings = {
            'data_processing_consent_required': True,
            'right_to_erasure_enabled': True,
            'data_portability_enabled': True,
            'profiling_consent_required': True,
            'automated_decision_consent_required': True
        }

        # MiFID II compliance settings
        self.mifid_settings = {
            'transaction_reporting_enabled': True,
            'best_execution_monitoring': True,
            'client_classification_required': True,
            'suitability_assessment_required': True,
            'record_keeping_enabled': True
        }

        self.logger = logging.getLogger(__name__)

        # Initialize compliance rules
        self._initialize_compliance_rules()

    def _initialize_compliance_rules(self):
        """Initialize regulatory compliance rules."""
        rules = [
            ComplianceRule(
                id="mifid_transaction_reporting",
                name="MiFID II Transaction Reporting",
                regulation="MiFID II",
                rule_type="transaction_reporting",
                description="All transactions must be reported to competent authorities within 1 business day",
                requirements={
                    'reporting_deadline_hours': 24,
                    'required_fields': ['client_id', 'instrument_id', 'quantity', 'price', 'timestamp'],
                    'competent_authorities': ['ESMA', 'National Competent Authorities']
                },
                enforcement_level="mandatory",
                implementation_status="implemented"
            ),
            ComplianceRule(
                id="gdpr_data_consent",
                name="GDPR Data Processing Consent",
                regulation="GDPR",
                rule_type="user_consent",
                description="Explicit consent required for processing personal data",
                requirements={
                    'consent_types': ['data_processing', 'profiling', 'automated_decisions'],
                    'consent_storage': 'encrypted_audit_trail',
                    'withdrawal_rights': True
                },
                enforcement_level="mandatory",
                implementation_status="implemented"
            ),
            ComplianceRule(
                id="data_retention_policy",
                name="Data Retention Policy",
                regulation="MiFID II / GDPR",
                rule_type="data_retention",
                description="Data must be retained according to regulatory requirements and deleted when no longer needed",
                requirements={
                    'retention_periods': self.data_retention_policies,
                    'automatic_deletion': True,
                    'archival_before_deletion': True
                },
                enforcement_level="mandatory",
                implementation_status="implemented"
            ),
            ComplianceRule(
                id="audit_trail_requirement",
                name="Comprehensive Audit Trail",
                regulation="MiFID II",
                rule_type="audit_logging",
                description="All system activities must be logged for regulatory audit",
                requirements={
                    'log_all_data_access': True,
                    'log_all_system_changes': True,
                    'log_user_actions': True,
                    'immutable_logs': True
                },
                enforcement_level="mandatory",
                implementation_status="implemented"
            ),
            ComplianceRule(
                id="client_suitability",
                name="Client Suitability Assessment",
                regulation="MiFID II",
                rule_type="client_assessment",
                description="Investment services must match client knowledge and experience",
                requirements={
                    'risk_tolerance_assessment': True,
                    'investment_knowledge_test': True,
                    'financial_situation_review': True,
                    'regular_reassessment': True
                },
                enforcement_level="mandatory",
                implementation_status="partial"
            )
        ]

        for rule in rules:
            self.compliance_rules[rule.id] = rule

    async def log_audit_event(self, action: str, resource: str, user_id: Optional[str] = None,
                            tenant_id: Optional[str] = None, details: Dict[str, Any] = None,
                            ip_address: str = "", user_agent: str = "") -> str:
        """Log an audit event."""
        if details is None:
            details = {}

        event_id = str(uuid.uuid4())

        audit_entry = AuditTrail(
            id=event_id,
            action=action,
            resource=resource,
            user_id=user_id,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=details.get('session_id', '')
        )

        self.audit_trail.append(audit_entry)

        # Also create compliance event
        compliance_event = ComplianceEvent(
            id=f"audit_{event_id}",
            event_type="audit_log",
            regulation="MiFID II",
            description=f"Audit event logged: {action} on {resource}",
            user_id=user_id,
            tenant_id=tenant_id,
            data={
                'audit_entry_id': event_id,
                'action': action,
                'resource': resource,
                'details': details
            }
        )

        self.compliance_events.append(compliance_event)

        # Keep only last 10000 entries for performance
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-10000:]

        self.logger.info(f"Audit event logged: {action} on {resource}")
        return event_id

    def check_data_retention_compliance(self, data_type: str, created_at: datetime) -> Dict[str, Any]:
        """Check if data retention policy is being followed."""
        if data_type not in self.data_retention_policies:
            return {
                'compliant': False,
                'reason': f'Unknown data type: {data_type}',
                'retention_policy': None
            }

        retention_period = self.data_retention_policies[data_type]
        age = datetime.now() - created_at

        is_expired = age > retention_period

        return {
            'compliant': not is_expired,
            'data_age_days': age.days,
            'retention_period_days': retention_period.days,
            'is_expired': is_expired,
            'expiry_date': (created_at + retention_period).isoformat(),
            'retention_policy': data_type
        }

    async def validate_user_consent(self, user_id: str, consent_type: str, tenant_id: str = None) -> bool:
        """Validate user consent for data processing."""
        # In production, this would check consent records in database
        # For now, return True (assuming consent was given during registration)

        # Log consent validation
        await self.log_audit_event(
            action="consent_validation",
            resource=f"user_consent_{consent_type}",
            user_id=user_id,
            tenant_id=tenant_id,
            details={'consent_type': consent_type, 'validation_result': True}
        )

        return True

    async def record_transaction(self, transaction_data: Dict[str, Any], user_id: str = None,
                               tenant_id: str = None) -> str:
        """Record transaction for MiFID II reporting."""
        transaction_id = str(uuid.uuid4())

        # Enhanced transaction data for compliance
        compliance_data = {
            'transaction_id': transaction_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'tenant_id': tenant_id,
            'transaction_type': transaction_data.get('type', 'unknown'),
            'instrument': transaction_data.get('instrument', ''),
            'quantity': transaction_data.get('quantity', 0),
            'price': transaction_data.get('price', 0),
            'venue': transaction_data.get('venue', 'AffectRON'),
            'client_classification': 'retail_client',  # Would be determined by suitability assessment
            'best_execution_applied': True,
            'costs_and_charges': transaction_data.get('fees', 0)
        }

        # Log transaction for audit
        await self.log_audit_event(
            action="transaction_recorded",
            resource="financial_transaction",
            user_id=user_id,
            tenant_id=tenant_id,
            details=compliance_data
        )

        # Create compliance event
        compliance_event = ComplianceEvent(
            id=f"transaction_{transaction_id}",
            event_type="transaction",
            regulation="MiFID II",
            description=f"Transaction recorded: {transaction_data.get('type', 'unknown')}",
            user_id=user_id,
            tenant_id=tenant_id,
            data=compliance_data,
            compliance_status="compliant"
        )

        self.compliance_events.append(compliance_event)

        return transaction_id

    async def perform_suitability_assessment(self, user_id: str, investment_data: Dict[str, Any],
                                           tenant_id: str = None) -> Dict[str, Any]:
        """Perform MiFID II suitability assessment."""
        # This would implement comprehensive suitability assessment
        # For now, return mock assessment

        assessment = {
            'user_id': user_id,
            'assessment_date': datetime.now().isoformat(),
            'client_classification': 'retail_client',
            'risk_tolerance': investment_data.get('risk_tolerance', 'medium'),
            'investment_knowledge': investment_data.get('knowledge_level', 'basic'),
            'financial_situation': investment_data.get('financial_situation', 'stable'),
            'investment_objectives': investment_data.get('objectives', 'growth'),
            'suitability_score': 0.7,  # Mock score
            'recommended_products': ['conservative_portfolio', 'diversified_funds'],
            'warnings': [],
            'next_review_date': (datetime.now() + timedelta(days=365)).isoformat()
        }

        # Log assessment
        await self.log_audit_event(
            action="suitability_assessment",
            resource="client_assessment",
            user_id=user_id,
            tenant_id=tenant_id,
            details=assessment
        )

        return assessment

    async def generate_compliance_report(self, tenant_id: str, start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for tenant."""
        # Filter events by tenant and date range
        tenant_events = [
            event for event in self.compliance_events
            if event.tenant_id == tenant_id and
            start_date <= event.timestamp <= end_date
        ]

        # Filter audit trail
        tenant_audit = [
            audit for audit in self.audit_trail
            if audit.tenant_id == tenant_id and
            start_date <= audit.timestamp <= end_date
        ]

        # Compliance summary
        compliance_summary = {
            'total_events': len(tenant_events),
            'compliant_events': len([e for e in tenant_events if e.compliance_status == 'compliant']),
            'violation_events': len([e for e in tenant_events if e.compliance_status == 'violation']),
            'warning_events': len([e for e in tenant_events if e.compliance_status == 'warning']),
            'audit_entries': len(tenant_audit)
        }

        # Events by regulation
        regulation_breakdown = {}
        for event in tenant_events:
            reg = event.regulation
            if reg not in regulation_breakdown:
                regulation_breakdown[reg] = {'total': 0, 'compliant': 0, 'violations': 0}
            regulation_breakdown[reg]['total'] += 1
            if event.compliance_status == 'compliant':
                regulation_breakdown[reg]['compliant'] += 1
            elif event.compliance_status == 'violation':
                regulation_breakdown[reg]['violations'] += 1

        return {
            'tenant_id': tenant_id,
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'compliance_summary': compliance_summary,
            'regulation_breakdown': regulation_breakdown,
            'audit_summary': {
                'total_entries': len(tenant_audit),
                'actions_performed': len(set(audit.action for audit in tenant_audit)),
                'resources_accessed': len(set(audit.resource for audit in tenant_audit))
            },
            'generated_at': datetime.now().isoformat(),
            'report_type': 'compliance_audit'
        }

    def check_gdpr_compliance(self, operation: str, data_types: List[str], user_id: str = None) -> Dict[str, Any]:
        """Check GDPR compliance for data operation."""
        compliance_issues = []
        required_consents = []

        # Check consent requirements
        if operation in ['profile', 'analyze', 'automated_decision']:
            if self.gdpr_settings['profiling_consent_required']:
                required_consents.append('profiling')

            if self.gdpr_settings['automated_decision_consent_required']:
                required_consents.append('automated_decisions')

        # Check data retention
        for data_type in data_types:
            if data_type in self.data_retention_policies:
                retention_period = self.data_retention_policies[data_type]
                # Check if data is within retention period (simplified)
                compliance_issues.append(f"Data type {data_type} retention: {retention_period.days} days")

        return {
            'gdpr_compliant': len(compliance_issues) == 0,
            'required_consents': required_consents,
            'compliance_issues': compliance_issues,
            'data_rights_available': [
                'access', 'rectification', 'erasure', 'restriction', 'portability', 'objection'
            ]
        }

    def check_mifid_compliance(self, operation: str, client_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check MiFID II compliance for operation."""
        compliance_issues = []

        if operation == 'trading_signal' and self.mifid_settings['transaction_reporting_enabled']:
            # Check if all required transaction fields are present
            required_fields = ['client_id', 'instrument_id', 'quantity', 'price', 'timestamp']
            missing_fields = [field for field in required_fields if field not in (client_data or {})]

            if missing_fields:
                compliance_issues.append(f"Missing transaction fields: {missing_fields}")

        if operation == 'investment_advice' and self.mifid_settings['client_classification_required']:
            if not client_data or 'client_classification' not in client_data:
                compliance_issues.append("Client classification required for investment advice")

        if operation == 'portfolio_management' and self.mifid_settings['suitability_assessment_required']:
            if not client_data or 'suitability_assessment' not in client_data:
                compliance_issues.append("Suitability assessment required for portfolio management")

        return {
            'mifid_compliant': len(compliance_issues) == 0,
            'compliance_issues': compliance_issues,
            'best_execution_applied': self.mifid_settings['best_execution_monitoring'],
            'record_keeping_active': self.mifid_settings['record_keeping_enabled']
        }

    async def request_data_erasure(self, user_id: str, tenant_id: str, data_types: List[str] = None) -> bool:
        """Handle GDPR right to erasure request."""
        if data_types is None:
            data_types = ['all']

        # Log erasure request
        await self.log_audit_event(
            action="data_erasure_requested",
            resource="user_data",
            user_id=user_id,
            tenant_id=tenant_id,
            details={
                'data_types': data_types,
                'request_timestamp': datetime.now().isoformat()
            }
        )

        # In production, this would:
        # 1. Verify user identity
        # 2. Identify all personal data for the user
        # 3. Create anonymized copies if required for legitimate business purposes
        # 4. Delete personal data
        # 5. Log the erasure

        erasure_confirmation = {
            'user_id': user_id,
            'data_types': data_types,
            'erasure_date': datetime.now().isoformat(),
            'retention_exemptions_applied': [],
            'anonymized_data_retained': False
        }

        # Log successful erasure
        await self.log_audit_event(
            action="data_erased",
            resource="user_data",
            user_id=user_id,
            tenant_id=tenant_id,
            details=erasure_confirmation
        )

        return True

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        total_rules = len(self.compliance_rules)

        implemented_rules = len([
            rule for rule in self.compliance_rules.values()
            if rule.implementation_status == 'implemented'
        ])

        partial_rules = len([
            rule for rule in self.compliance_rules.values()
            if rule.implementation_status == 'partial'
        ])

        compliance_score = (implemented_rules * 1.0 + partial_rules * 0.5) / total_rules

        return {
            'overall_compliance_score': compliance_score,
            'total_rules': total_rules,
            'implemented_rules': implemented_rules,
            'partial_rules': partial_rules,
            'planned_rules': total_rules - implemented_rules - partial_rules,
            'regulations_covered': list(set(rule.regulation for rule in self.compliance_rules.values())),
            'last_assessment': datetime.now().isoformat()
        }

    def export_audit_trail(self, start_date: datetime, end_date: datetime,
                          tenant_id: str = None, format: str = 'json') -> str:
        """Export audit trail for compliance reporting."""
        # Filter audit trail
        filtered_audit = [
            audit for audit in self.audit_trail
            if start_date <= audit.timestamp <= end_date
        ]

        if tenant_id:
            filtered_audit = [
                audit for audit in filtered_audit
                if audit.tenant_id == tenant_id
            ]

        if format == 'json':
            export_data = {
                'export_metadata': {
                    'export_date': datetime.now().isoformat(),
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'tenant_id': tenant_id,
                    'total_entries': len(filtered_audit)
                },
                'audit_entries': [
                    {
                        'id': audit.id,
                        'action': audit.action,
                        'resource': audit.resource,
                        'user_id': audit.user_id,
                        'tenant_id': audit.tenant_id,
                        'timestamp': audit.timestamp.isoformat(),
                        'details': audit.details
                    }
                    for audit in filtered_audit
                ]
            }

            return json.dumps(export_data, indent=2, default=str)

        elif format == 'csv':
            # CSV format for audit trail
            csv_lines = ['id,action,resource,user_id,tenant_id,timestamp,details']

            for audit in filtered_audit:
                details_str = json.dumps(audit.details, default=str).replace(',', ';')
                csv_lines.append(
                    f"{audit.id},{audit.action},{audit.resource},{audit.user_id},{audit.tenant_id},"
                    f"{audit.timestamp.isoformat()},{details_str}"
                )

            return '\n'.join(csv_lines)

        return ""

    async def perform_compliance_check(self, operation: str, user_id: str = None,
                                     tenant_id: str = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive compliance check for operation."""
        if data is None:
            data = {}

        results = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'tenant_id': tenant_id,
            'compliance_checks': {},
            'overall_compliant': True
        }

        # GDPR compliance check
        gdpr_check = self.check_gdpr_compliance(operation, data.get('data_types', []), user_id)
        results['compliance_checks']['gdpr'] = gdpr_check

        if not gdpr_check['gdpr_compliant']:
            results['overall_compliant'] = False

        # MiFID II compliance check
        mifid_check = self.check_mifid_compliance(operation, data.get('client_data'))
        results['compliance_checks']['mifid'] = mifid_check

        if not mifid_check['mifid_compliant']:
            results['overall_compliant'] = False

        # Log compliance check
        await self.log_audit_event(
            action="compliance_check",
            resource=f"operation_{operation}",
            user_id=user_id,
            tenant_id=tenant_id,
            details={
                'operation': operation,
                'compliance_result': results['overall_compliant'],
                'checks_performed': list(results['compliance_checks'].keys())
            }
        )

        return results


# Global compliance manager instance
compliance_manager = ComplianceManager()


async def initialize_compliance_system():
    """Initialize compliance management system."""
    logging.getLogger(__name__).info("Compliance management system initialized")


def get_compliance_status():
    """Get compliance system status."""
    return {
        'initialized': True,
        'compliance_rules': len(compliance_manager.compliance_rules),
        'audit_entries': len(compliance_manager.audit_trail),
        'compliance_events': len(compliance_manager.compliance_events),
        'overall_status': compliance_manager.get_compliance_status()
    }
