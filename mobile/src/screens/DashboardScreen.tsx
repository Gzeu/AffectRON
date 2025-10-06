import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Dimensions,
} from 'react-native';
import {Card, Title, Paragraph} from 'react-native-paper';
import {LineChart} from 'react-native-chart-kit';
import axios from 'axios';

const {width} = Dimensions.get('window');

interface DashboardData {
  overall_sentiment: {
    label: string;
    score: number;
  };
  market_health: {
    score: number;
    status: string;
  };
  risk_level: {
    level: string;
    score: number;
  };
  active_alerts: number;
  data_sources_status: {
    total: number;
    active: number;
  };
}

const DashboardScreen: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      // In production, this would connect to your API
      const response = await axios.get('http://localhost:8000/api/v1/dashboard');

      if (response.data) {
        setDashboardData(response.data);
      } else {
        // Mock data for development
        setDashboardData({
          overall_sentiment: {
            label: 'Positive',
            score: 0.7,
          },
          market_health: {
            score: 0.8,
            status: 'Good',
          },
          risk_level: {
            level: 'Medium',
            score: 0.4,
          },
          active_alerts: 2,
          data_sources_status: {
            total: 5,
            active: 4,
          },
        });
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      // Set mock data on error
      setDashboardData({
        overall_sentiment: {
          label: 'Positive',
          score: 0.7,
        },
        market_health: {
          score: 0.8,
          status: 'Good',
        },
        risk_level: {
          level: 'Medium',
          score: 0.4,
        },
        active_alerts: 2,
        data_sources_status: {
          total: 5,
          active: 4,
        },
      });
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchDashboardData();
    setRefreshing(false);
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.6) return '#4caf50'; // Green for positive
    if (score < 0.4) return '#f44336'; // Red for negative
    return '#ff9800'; // Orange for neutral
  };

  const getRiskColor = (score: number) => {
    if (score < 0.3) return '#4caf50'; // Green for low risk
    if (score < 0.7) return '#ff9800'; // Orange for medium risk
    return '#f44336'; // Red for high risk
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <Text>Loading dashboard...</Text>
      </View>
    );
  }

  if (!dashboardData) {
    return (
      <View style={styles.centerContainer}>
        <Text>Error loading dashboard data</Text>
      </View>
    );
  }

  const chartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    datasets: [
      {
        data: [0.5, 0.7, 0.6, 0.8, 0.7],
        strokeWidth: 2,
      },
    ],
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }>
      <View style={styles.header}>
        <Text style={styles.title}>AffectRON Mobile</Text>
        <Text style={styles.subtitle}>Financial Sentiment Analysis</Text>
      </View>

      {/* Overall Sentiment Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Sentiment General</Title>
          <View style={styles.metricContainer}>
            <Text style={[styles.metricValue, {color: getSentimentColor(dashboardData.overall_sentiment.score)}]}>
              {dashboardData.overall_sentiment.label}
            </Text>
            <Text style={styles.metricScore}>
              {dashboardData.overall_sentiment.score.toFixed(2)}
            </Text>
          </View>
        </Card.Content>
      </Card>

      {/* Market Health Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Sănătatea Pieței</Title>
          <View style={styles.metricContainer}>
            <Text style={[styles.metricValue, {color: getSentimentColor(dashboardData.market_health.score)}]}>
              {dashboardData.market_health.status}
            </Text>
            <Text style={styles.metricScore}>
              {dashboardData.market_health.score.toFixed(2)}
            </Text>
          </View>
        </Card.Content>
      </Card>

      {/* Risk Level Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Nivel de Risc</Title>
          <View style={styles.metricContainer}>
            <Text style={[styles.metricValue, {color: getRiskColor(dashboardData.risk_level.score)}]}>
              {dashboardData.risk_level.level}
            </Text>
            <Text style={styles.metricScore}>
              {dashboardData.risk_level.score.toFixed(2)}
            </Text>
          </View>
        </Card.Content>
      </Card>

      {/* Active Alerts Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Alerte Active</Title>
          <Text style={styles.alertCount}>{dashboardData.active_alerts}</Text>
          <Text style={styles.alertText}>
            {dashboardData.active_alerts === 0 ? 'Nicio alertă' : 'Alerte care necesită atenție'}
          </Text>
        </Card.Content>
      </Card>

      {/* Data Sources Status Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Surse de Date</Title>
          <Text style={styles.sourceStatus}>
            {dashboardData.data_sources_status.active}/{dashboardData.data_sources_status.total} active
          </Text>
          <Text style={styles.sourceText}>
            Starea surselor de date în timp real
          </Text>
        </Card.Content>
      </Card>

      {/* Sentiment Trend Chart */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Trend Sentiment</Title>
          <LineChart
            data={chartData}
            width={width - 40}
            height={220}
            chartConfig={{
              backgroundColor: '#ffffff',
              backgroundGradientFrom: '#f0f0f0',
              backgroundGradientTo: '#ffffff',
              decimalPlaces: 2,
              color: (opacity = 1) => `rgba(25, 118, 210, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
            }}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>

      {/* Last Updated */}
      <View style={styles.footer}>
        <Text style={styles.lastUpdated}>
          Ultima actualizare: {new Date().toLocaleTimeString('ro-RO')}
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#1976d2',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  subtitle: {
    fontSize: 14,
    color: '#e3f2fd',
    marginTop: 4,
  },
  card: {
    margin: 10,
    elevation: 4,
  },
  metricContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 10,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  metricScore: {
    fontSize: 16,
    color: '#666',
  },
  alertCount: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#f44336',
    textAlign: 'center',
  },
  alertText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  sourceStatus: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4caf50',
    textAlign: 'center',
  },
  sourceText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 8,
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  lastUpdated: {
    fontSize: 12,
    color: '#666',
  },
});

export default DashboardScreen;
