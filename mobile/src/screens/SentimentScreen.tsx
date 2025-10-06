import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import {Card, Title, Paragraph, Button} from 'react-native-paper';
import {LineChart, PieChart} from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialIcons';

const {width} = Dimensions.get('window');

interface SentimentData {
  currency: string;
  sentiment: {
    label: string;
    score: number;
    confidence: number;
  };
  trend: number[];
  entities: {
    currencies: string[];
    institutions: string[];
  };
}

const SentimentScreen: React.FC = () => {
  const [selectedCurrency, setSelectedCurrency] = useState('RON');
  const [customText, setCustomText] = useState('');
  const [sentimentResult, setSentimentResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const currencies = [
    {code: 'RON', name: 'Leu Românesc'},
    {code: 'EUR', name: 'Euro'},
    {code: 'USD', name: 'Dolar American'},
    {code: 'GBP', name: 'Liră Sterlină'},
    {code: 'CHF', name: 'Franc Elvețian'},
  ];

  // Mock sentiment data for different currencies
  const mockSentimentData: Record<string, SentimentData> = {
    RON: {
      currency: 'RON',
      sentiment: {
        label: 'Positive',
        score: 0.7,
        confidence: 0.85,
      },
      trend: [0.5, 0.6, 0.7, 0.65, 0.7, 0.75, 0.7],
      entities: {
        currencies: ['RON', 'EUR'],
        institutions: ['BNR'],
      },
    },
    EUR: {
      currency: 'EUR',
      sentiment: {
        label: 'Neutral',
        score: 0.5,
        confidence: 0.75,
      },
      trend: [0.4, 0.5, 0.55, 0.5, 0.45, 0.5, 0.52],
      entities: {
        currencies: ['EUR', 'USD'],
        institutions: ['ECB'],
      },
    },
    USD: {
      currency: 'USD',
      sentiment: {
        label: 'Negative',
        score: 0.3,
        confidence: 0.8,
      },
      trend: [0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.3],
      entities: {
        currencies: ['USD', 'EUR'],
        institutions: ['FED'],
      },
    },
  };

  const [sentimentData, setSentimentData] = useState<SentimentData>(
    mockSentimentData[selectedCurrency],
  );

  useEffect(() => {
    setSentimentData(mockSentimentData[selectedCurrency]);
  }, [selectedCurrency]);

  const analyzeCustomText = async () => {
    if (!customText.trim()) return;

    setLoading(true);
    try {
      // In production, this would call your API
      // const response = await axios.post('http://localhost:8000/api/v1/sentiment/analyze', {
      //   text: customText,
      //   language: 'ro'
      // });

      // Mock response
      setTimeout(() => {
        setSentimentResult({
          text: customText,
          sentiment: {
            label: Math.random() > 0.5 ? 'Positive' : 'Negative',
            score: Math.random(),
            confidence: Math.random(),
          },
          entities: {
            currencies: ['RON'],
            institutions: [],
          },
        });
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error analyzing text:', error);
      setLoading(false);
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.6) return '#4caf50'; // Green
    if (score < 0.4) return '#f44336'; // Red
    return '#ff9800'; // Orange
  };

  const getSentimentIcon = (label: string) => {
    switch (label.toLowerCase()) {
      case 'positive':
        return 'sentiment-satisfied';
      case 'negative':
        return 'sentiment-dissatisfied';
      default:
        return 'sentiment-neutral';
    }
  };

  const renderCurrencySelector = () => (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.currencySelector}>
      {currencies.map((currency) => (
        <TouchableOpacity
          key={currency.code}
          style={[
            styles.currencyButton,
            selectedCurrency === currency.code && styles.currencyButtonSelected,
          ]}
          onPress={() => setSelectedCurrency(currency.code)}>
          <Text
            style={[
              styles.currencyButtonText,
              selectedCurrency === currency.code && styles.currencyButtonTextSelected,
            ]}>
            {currency.code}
          </Text>
          <Text style={styles.currencyName}>{currency.name}</Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );

  const renderSentimentCard = () => {
    const sentiment = sentimentData.sentiment;

    return (
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Icon name={getSentimentIcon(sentiment.label)} size={24} color={getSentimentColor(sentiment.score)} />
            <Title style={styles.cardTitle}>Analiză Sentiment {selectedCurrency}</Title>
          </View>

          <View style={styles.sentimentContainer}>
            <Text style={[styles.sentimentLabel, {color: getSentimentColor(sentiment.score)}]}>
              {sentiment.label}
            </Text>
            <Text style={styles.sentimentScore}>{sentiment.score.toFixed(3)}</Text>
            <Text style={styles.confidenceText}>
              Încredere: {(sentiment.confidence * 100).toFixed(1)}%
            </Text>
          </View>

          {/* Sentiment Trend Chart */}
          <View style={styles.chartContainer}>
            <Text style={styles.chartTitle}>Trend Sentiment (7 zile)</Text>
            <LineChart
              data={{
                labels: ['L', 'M', 'M', 'J', 'V', 'S', 'D'],
                datasets: [{data: sentimentData.trend, strokeWidth: 2}],
              }}
              width={width - 60}
              height={180}
              chartConfig={{
                backgroundColor: '#ffffff',
                backgroundGradientFrom: '#f8f9fa',
                backgroundGradientTo: '#ffffff',
                decimalPlaces: 2,
                color: (opacity = 1) => `rgba(25, 118, 210, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              }}
              bezier
              style={styles.chart}
            />
          </View>
        </Card.Content>
      </Card>
    );
  };

  const renderEntitiesCard = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title>Entități Detectate</Title>

        {sentimentData.entities.currencies.length > 0 && (
          <View style={styles.entitiesSection}>
            <Text style={styles.entitiesTitle}>Monede:</Text>
            <View style={styles.entitiesContainer}>
              {sentimentData.entities.currencies.map((currency, index) => (
                <View key={index} style={styles.entityChip}>
                  <Text style={styles.entityText}>{currency}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {sentimentData.entities.institutions.length > 0 && (
          <View style={styles.entitiesSection}>
            <Text style={styles.entitiesTitle}>Instituții:</Text>
            <View style={styles.entitiesContainer}>
              {sentimentData.entities.institutions.map((institution, index) => (
                <View key={index} style={styles.entityChip}>
                  <Text style={styles.entityText}>{institution}</Text>
                </View>
              ))}
            </View>
          </View>
        )}
      </Card.Content>
    </Card>
  );

  const renderCustomAnalysisCard = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title>Analiză Text Personalizat</Title>

        <TextInput
          style={styles.textInput}
          multiline
          placeholder="Introduceți text pentru analiză..."
          value={customText}
          onChangeText={setCustomText}
          textAlignVertical="top"
        />

        <Button
          mode="contained"
          onPress={analyzeCustomText}
          loading={loading}
          disabled={!customText.trim() || loading}
          style={styles.analyzeButton}>
          Analizează
        </Button>

        {sentimentResult && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Rezultat Analiză:</Text>
            <Text style={[styles.resultSentiment, {color: getSentimentColor(sentimentResult.sentiment.score)}]}>
              {sentimentResult.sentiment.label} ({sentimentResult.sentiment.score.toFixed(3)})
            </Text>
            <Text style={styles.resultText}>
              Text: {sentimentResult.text.substring(0, 100)}...
            </Text>
          </View>
        )}
      </Card.Content>
    </Card>
  );

  const renderSentimentDistribution = () => {
    const pieData = [
      {
        name: 'Pozitiv',
        value: 45,
        color: '#4caf50',
        legendFontColor: '#7F7F7F',
        legendFontSize: 12,
      },
      {
        name: 'Neutru',
        value: 35,
        color: '#ff9800',
        legendFontColor: '#7F7F7F',
        legendFontSize: 12,
      },
      {
        name: 'Negativ',
        value: 20,
        color: '#f44336',
        legendFontColor: '#7F7F7F',
        legendFontSize: 12,
      },
    ];

    return (
      <Card style={styles.card}>
        <Card.Content>
          <Title>Distribuție Sentiment</Title>
          <PieChart
            data={pieData}
            width={width - 40}
            height={200}
            chartConfig={{
              backgroundColor: '#ffffff',
              backgroundGradientFrom: '#f8f9fa',
              backgroundGradientTo: '#ffffff',
              color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
            }}
            accessor="value"
            backgroundColor="transparent"
            paddingLeft="15"
            center={[10, 10]}
          />
        </Card.Content>
      </Card>
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Currency Selector */}
      {renderCurrencySelector()}

      {/* Sentiment Analysis Card */}
      {renderSentimentCard()}

      {/* Entities Card */}
      {renderEntitiesCard()}

      {/* Sentiment Distribution */}
      {renderSentimentDistribution()}

      {/* Custom Text Analysis */}
      {renderCustomAnalysisCard()}

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Analiza sentiment în timp real pentru {selectedCurrency}
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
  currencySelector: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  currencyButton: {
    backgroundColor: '#e0e0e0',
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginRight: 10,
    borderRadius: 20,
    alignItems: 'center',
    minWidth: 80,
  },
  currencyButtonSelected: {
    backgroundColor: '#1976d2',
  },
  currencyButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  currencyButtonTextSelected: {
    color: '#ffffff',
  },
  currencyName: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  card: {
    margin: 10,
    elevation: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  cardTitle: {
    marginLeft: 10,
    fontSize: 18,
  },
  sentimentContainer: {
    alignItems: 'center',
    marginVertical: 15,
  },
  sentimentLabel: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  sentimentScore: {
    fontSize: 18,
    color: '#666',
    marginTop: 5,
  },
  confidenceText: {
    fontSize: 14,
    color: '#888',
    marginTop: 2,
  },
  chartContainer: {
    marginTop: 15,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 8,
  },
  entitiesSection: {
    marginTop: 15,
  },
  entitiesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  entitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  entityChip: {
    backgroundColor: '#e3f2fd',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 6,
    marginBottom: 6,
  },
  entityText: {
    fontSize: 12,
    color: '#1976d2',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    minHeight: 100,
    backgroundColor: '#ffffff',
    marginVertical: 10,
  },
  analyzeButton: {
    marginVertical: 10,
  },
  resultContainer: {
    marginTop: 15,
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resultSentiment: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  resultText: {
    fontSize: 14,
    color: '#666',
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});

export default SentimentScreen;
