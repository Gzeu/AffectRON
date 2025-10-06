import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
} from 'react-native';
import {Card, Title, Paragraph, Chip, Button} from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialIcons';

interface Alert {
  id: string;
  title: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  timestamp: string;
  acknowledged: boolean;
  data?: any;
}

const AlertsScreen: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'high_priority'>('all');

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    try {
      // In production, this would connect to your API
      // const response = await axios.get('http://localhost:8000/api/v1/alerts/active');

      // Mock data for development
      const mockAlerts: Alert[] = [
        {
          id: 'alert_1',
          title: 'High Sentiment Volatility Detected - RON',
          message: 'Sentiment volatility of 0.85 exceeds threshold of 0.30 for RON',
          severity: 'high',
          type: 'sentiment_spike',
          timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 minutes ago
          acknowledged: false,
          data: {
            currency: 'RON',
            volatility: 0.85,
          },
        },
        {
          id: 'alert_2',
          title: 'Market Volatility Alert - EUR',
          message: 'Price change of 2.5% exceeds threshold for EUR',
          severity: 'medium',
          type: 'market_volatility',
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
          acknowledged: true,
          data: {
            currency: 'EUR',
            price_change: 0.025,
          },
        },
        {
          id: 'alert_3',
          title: 'Data Source Error - Twitter API',
          message: 'Twitter API has 3 consecutive errors',
          severity: 'low',
          type: 'data_source_error',
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4).toISOString(), // 4 hours ago
          acknowledged: false,
          data: {
            source_name: 'Twitter API',
            error_count: 3,
          },
        },
      ];

      setAlerts(mockAlerts);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchAlerts();
    setRefreshing(false);
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      // In production, this would call your API
      // await axios.post(`http://localhost:8000/api/v1/alerts/${alertId}/acknowledge`);

      // Update local state
      setAlerts(prevAlerts =>
        prevAlerts.map(alert =>
          alert.id === alertId
            ? {...alert, acknowledged: true}
            : alert,
        ),
      );
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#d32f2f'; // Red
      case 'high':
        return '#f57c00'; // Orange
      case 'medium':
        return '#fbc02d'; // Yellow
      case 'low':
        return '#388e3c'; // Green
      default:
        return '#757575'; // Gray
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'check-circle';
      default:
        return 'notifications';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    switch (filter) {
      case 'unacknowledged':
        return !alert.acknowledged;
      case 'high_priority':
        return alert.severity === 'high' || alert.severity === 'critical';
      default:
        return true;
    }
  });

  const renderFilterButtons = () => (
    <View style={styles.filterContainer}>
      <TouchableOpacity
        style={[styles.filterButton, filter === 'all' && styles.filterButtonActive]}
        onPress={() => setFilter('all')}>
        <Text style={[styles.filterButtonText, filter === 'all' && styles.filterButtonTextActive]}>
          Toate
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.filterButton, filter === 'unacknowledged' && styles.filterButtonActive]}
        onPress={() => setFilter('unacknowledged')}>
        <Text style={[styles.filterButtonText, filter === 'unacknowledged' && styles.filterButtonTextActive]}>
          Nealocate
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.filterButton, filter === 'high_priority' && styles.filterButtonActive]}
        onPress={() => setFilter('high_priority')}>
        <Text style={[styles.filterButtonText, filter === 'high_priority' && styles.filterButtonTextActive]}>
          Prioritate
        </Text>
      </TouchableOpacity>
    </View>
  );

  const renderAlertCard = (alert: Alert) => (
    <Card key={alert.id} style={[styles.alertCard, !alert.acknowledged && styles.unacknowledgedAlert]}>
      <Card.Content>
        <View style={styles.alertHeader}>
          <View style={styles.alertTitleContainer}>
            <Icon
              name={getSeverityIcon(alert.severity)}
              size={20}
              color={getSeverityColor(alert.severity)}
            />
            <Title style={styles.alertTitle}>{alert.title}</Title>
          </View>

          <Chip
            style={[styles.severityChip, {backgroundColor: getSeverityColor(alert.severity)}]}>
            <Text style={styles.severityChipText}>{alert.severity.toUpperCase()}</Text>
          </Chip>
        </View>

        <Paragraph style={styles.alertMessage}>{alert.message}</Paragraph>

        <View style={styles.alertMeta}>
          <Text style={styles.alertType}>{alert.type.replace('_', ' ').toUpperCase()}</Text>
          <Text style={styles.alertTimestamp}>
            {new Date(alert.timestamp).toLocaleString('ro-RO')}
          </Text>
        </View>

        {alert.data && (
          <View style={styles.alertData}>
            {alert.data.currency && (
              <Text style={styles.alertDataText}>Monedă: {alert.data.currency}</Text>
            )}
            {alert.data.volatility && (
              <Text style={styles.alertDataText}>Volatilitate: {alert.data.volatility.toFixed(3)}</Text>
            )}
            {alert.data.price_change && (
              <Text style={styles.alertDataText}>
                Schimbare preț: {(alert.data.price_change * 100).toFixed(2)}%
              </Text>
            )}
          </View>
        )}

        {!alert.acknowledged && (
          <Button
            mode="contained"
            onPress={() => acknowledgeAlert(alert.id)}
            style={styles.acknowledgeButton}>
            Marchează ca citit
          </Button>
        )}
      </Card.Content>
    </Card>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Alerte și Notificări</Text>
        <Text style={styles.headerSubtitle}>
          {filteredAlerts.length} alerte {filter !== 'all' ? `(${filter})` : ''}
        </Text>
      </View>

      {/* Filter Buttons */}
      {renderFilterButtons()}

      {/* Alerts List */}
      <ScrollView
        style={styles.alertsContainer}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        {filteredAlerts.length === 0 ? (
          <View style={styles.emptyState}>
            <Icon name="notifications-none" size={48} color="#ccc" />
            <Text style={styles.emptyStateText}>
              {filter === 'unacknowledged'
                ? 'Nicio alertă nealocată'
                : filter === 'high_priority'
                ? 'Nicio alertă de prioritate înaltă'
                : 'Nicio alertă'}
            </Text>
          </View>
        ) : (
          filteredAlerts.map(renderAlertCard)
        )}

        {/* Spacer for bottom */}
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#1976d2',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#e3f2fd',
    marginTop: 4,
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#ffffff',
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  filterButtonActive: {
    backgroundColor: '#1976d2',
  },
  filterButtonText: {
    fontSize: 14,
    color: '#666',
  },
  filterButtonTextActive: {
    color: '#ffffff',
  },
  alertsContainer: {
    flex: 1,
  },
  alertCard: {
    margin: 10,
    elevation: 2,
  },
  unacknowledgedAlert: {
    borderLeftWidth: 4,
    borderLeftColor: '#ff9800',
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  alertTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    marginRight: 8,
  },
  alertTitle: {
    fontSize: 16,
    marginLeft: 8,
    flex: 1,
  },
  severityChip: {
    height: 24,
  },
  severityChipText: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  alertMessage: {
    fontSize: 14,
    marginBottom: 8,
    lineHeight: 20,
  },
  alertMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  alertType: {
    fontSize: 12,
    color: '#666',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  alertTimestamp: {
    fontSize: 12,
    color: '#999',
  },
  alertData: {
    backgroundColor: '#f8f9fa',
    padding: 8,
    borderRadius: 4,
    marginBottom: 8,
  },
  alertDataText: {
    fontSize: 12,
    color: '#555',
    marginBottom: 2,
  },
  acknowledgeButton: {
    marginTop: 8,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#999',
    marginTop: 16,
  },
  bottomSpacer: {
    height: 20,
  },
});

export default AlertsScreen;
