import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import {Card, Title, DataTable} from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialIcons';

const {width} = Dimensions.get('window');

interface MarketData {
  pair: string;
  rate: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

const MarketScreen: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '24h' | '7d'>('24h');

  useEffect(() => {
    fetchMarketData();
  }, [selectedTimeframe]);

  const fetchMarketData = async () => {
    try {
      // In production, this would connect to your API
      // const response = await axios.get(`http://localhost:8000/api/v1/market/rates?timeframe=${selectedTimeframe}`);

      // Mock data for development
      const mockData: MarketData[] = [
        {
          pair: 'EUR/RON',
          rate: 4.9750,
          change: 0.0025,
          changePercent: 0.05,
          volume: 1500000000,
          timestamp: new Date().toISOString(),
        },
        {
          pair: 'USD/RON',
          rate: 4.5800,
          change: -0.0080,
          changePercent: -0.17,
          volume: 1200000000,
          timestamp: new Date().toISOString(),
        },
        {
          pair: 'GBP/RON',
          rate: 5.8500,
          change: 0.0150,
          changePercent: 0.26,
          volume: 800000000,
          timestamp: new Date().toISOString(),
        },
        {
          pair: 'CHF/RON',
          rate: 5.2000,
          change: 0.0050,
          changePercent: 0.10,
          volume: 600000000,
          timestamp: new Date().toISOString(),
        },
      ];

      setMarketData(mockData);
    } catch (error) {
      console.error('Error fetching market data:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchMarketData();
    setRefreshing(false);
  };

  const formatNumber = (num: number, decimals: number = 4) => {
    return num.toFixed(decimals);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000000) {
      return `${(volume / 1000000000).toFixed(1)}B`;
    } else if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else {
      return `${(volume / 1000).toFixed(1)}K`;
    }
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return '#4caf50'; // Green for positive
    if (change < 0) return '#f44336'; // Red for negative
    return '#666'; // Gray for neutral
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return 'trending-up';
    if (change < 0) return 'trending-down';
    return 'trending-flat';
  };

  const renderTimeframeSelector = () => (
    <View style={styles.timeframeContainer}>
      {(['1h', '24h', '7d'] as const).map((timeframe) => (
        <TouchableOpacity
          key={timeframe}
          style={[
            styles.timeframeButton,
            selectedTimeframe === timeframe && styles.timeframeButtonActive,
          ]}
          onPress={() => setSelectedTimeframe(timeframe)}>
          <Text
            style={[
              styles.timeframeButtonText,
              selectedTimeframe === timeframe && styles.timeframeButtonTextActive,
            ]}>
            {timeframe === '1h' ? '1H' : timeframe === '24h' ? '24H' : '7D'}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderMarketTable = () => (
    <Card style={styles.tableCard}>
      <Card.Content>
        <Title style={styles.tableTitle}>Rate de Schimb ({selectedTimeframe.toUpperCase()})</Title>

        <DataTable>
          <DataTable.Header>
            <DataTable.Title style={styles.pairColumn}>Pereche</DataTable.Title>
            <DataTable.Title numeric style={styles.rateColumn}>Curs</DataTable.Title>
            <DataTable.Title numeric style={styles.changeColumn}>Schimbare</DataTable.Title>
            <DataTable.Title numeric style={styles.volumeColumn}>Volum</DataTable.Title>
          </DataTable.Header>

          {marketData.map((item, index) => (
            <DataTable.Row key={index}>
              <DataTable.Cell style={styles.pairColumn}>
                <Text style={styles.pairText}>{item.pair}</Text>
              </DataTable.Cell>
              <DataTable.Cell numeric style={styles.rateColumn}>
                <Text style={styles.rateText}>{formatNumber(item.rate)}</Text>
              </DataTable.Cell>
              <DataTable.Cell numeric style={styles.changeColumn}>
                <View style={styles.changeContainer}>
                  <Icon
                    name={getChangeIcon(item.change)}
                    size={16}
                    color={getChangeColor(item.change)}
                  />
                  <Text style={[styles.changeText, {color: getChangeColor(item.change)}]}>
                    {item.change >= 0 ? '+' : ''}{formatNumber(item.changePercent, 2)}%
                  </Text>
                </View>
              </DataTable.Cell>
              <DataTable.Cell numeric style={styles.volumeColumn}>
                <Text style={styles.volumeText}>{formatVolume(item.volume)}</Text>
              </DataTable.Cell>
            </DataTable.Row>
          ))}
        </DataTable>
      </Card.Content>
    </Card>
  );

  const renderMarketSummary = () => {
    const positiveChanges = marketData.filter(item => item.change > 0).length;
    const negativeChanges = marketData.filter(item => item.change < 0).length;

    return (
      <Card style={styles.summaryCard}>
        <Card.Content>
          <Title>Rezumat Piață</Title>

          <View style={styles.summaryContainer}>
            <View style={styles.summaryItem}>
              <Icon name="trending-up" size={20} color="#4caf50" />
              <Text style={styles.summaryValue}>{positiveChanges}</Text>
              <Text style={styles.summaryLabel}>În creștere</Text>
            </View>

            <View style={styles.summaryItem}>
              <Icon name="trending-down" size={20} color="#f44336" />
              <Text style={styles.summaryValue}>{negativeChanges}</Text>
              <Text style={styles.summaryLabel}>În scădere</Text>
            </View>

            <View style={styles.summaryItem}>
              <Icon name="show-chart" size={20} color="#2196f3" />
              <Text style={styles.summaryValue}>{marketData.length}</Text>
              <Text style={styles.summaryLabel}>Total perechi</Text>
            </View>
          </View>
        </Card.Content>
      </Card>
    );
  };

  const renderTopMovers = () => {
    // Sort by absolute change percentage
    const sortedData = [...marketData].sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent));
    const topMovers = sortedData.slice(0, 3);

    return (
      <Card style={styles.card}>
        <Card.Content>
          <Title>Mișcări Importante</Title>

          {topMovers.map((item, index) => (
            <View key={index} style={styles.moverItem}>
              <View style={styles.moverHeader}>
                <Text style={styles.moverPair}>{item.pair}</Text>
                <View style={styles.moverChange}>
                  <Icon
                    name={getChangeIcon(item.change)}
                    size={16}
                    color={getChangeColor(item.change)}
                  />
                  <Text style={[styles.moverChangeText, {color: getChangeColor(item.change)}]}>
                    {item.change >= 0 ? '+' : ''}{formatNumber(item.changePercent, 2)}%
                  </Text>
                </View>
              </View>
              <Text style={styles.moverRate}>{formatNumber(item.rate)}</Text>
            </View>
          ))}
        </Card.Content>
      </Card>
    );
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Piață și Rate</Text>
        <Text style={styles.headerSubtitle}>
          Date în timp real pentru {selectedTimeframe}
        </Text>
      </View>

      {/* Timeframe Selector */}
      {renderTimeframeSelector()}

      {/* Market Summary */}
      {renderMarketSummary()}

      {/* Market Table */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.tableScroll}>
        {renderMarketTable()}
      </ScrollView>

      {/* Top Movers */}
      {renderTopMovers()}

      {/* Refresh indicator */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }>
        <View style={styles.lastUpdated}>
          <Text style={styles.lastUpdatedText}>
            Ultima actualizare: {new Date().toLocaleTimeString('ro-RO')}
          </Text>
        </View>
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
  timeframeContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#ffffff',
  },
  timeframeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  timeframeButtonActive: {
    backgroundColor: '#1976d2',
  },
  timeframeButtonText: {
    fontSize: 14,
    color: '#666',
  },
  timeframeButtonTextActive: {
    color: '#ffffff',
  },
  summaryCard: {
    margin: 10,
    elevation: 2,
  },
  summaryContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 15,
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 4,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  tableScroll: {
    maxHeight: 300,
  },
  tableCard: {
    margin: 10,
    elevation: 2,
  },
  tableTitle: {
    fontSize: 16,
    marginBottom: 10,
  },
  pairColumn: {
    flex: 2,
  },
  rateColumn: {
    flex: 2,
  },
  changeColumn: {
    flex: 2,
  },
  volumeColumn: {
    flex: 2,
  },
  pairText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  rateText: {
    fontSize: 14,
    color: '#333',
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  changeText: {
    fontSize: 12,
    marginLeft: 4,
  },
  volumeText: {
    fontSize: 12,
    color: '#666',
  },
  card: {
    margin: 10,
    elevation: 2,
  },
  moverItem: {
    marginVertical: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  moverHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  moverPair: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  moverChange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  moverChangeText: {
    fontSize: 14,
    marginLeft: 4,
  },
  moverRate: {
    fontSize: 12,
    color: '#666',
  },
  content: {
    flex: 1,
  },
  lastUpdated: {
    padding: 20,
    alignItems: 'center',
  },
  lastUpdatedText: {
    fontSize: 12,
    color: '#666',
  },
});

export default MarketScreen;
