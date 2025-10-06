import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';

import SentimentChart from '../components/SentimentChart';
import EntityList from '../components/EntityList';

const SentimentAnalysis: React.FC = () => {
  const [selectedCurrency, setSelectedCurrency] = useState('RON');
  const [timeframe, setTimeframe] = useState('24h');
  const [customText, setCustomText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  const currencies = [
    { value: 'RON', label: 'RON (Leu Românesc)' },
    { value: 'EUR', label: 'EUR (Euro)' },
    { value: 'USD', label: 'USD (Dolar American)' },
    { value: 'BTC', label: 'BTC (Bitcoin)' },
    { value: 'ETH', label: 'ETH (Ethereum)' },
  ];

  const timeframes = [
    { value: '1h', label: 'Ultima oră' },
    { value: '6h', label: 'Ultimele 6 ore' },
    { value: '24h', label: 'Ultimele 24 ore' },
    { value: '7d', label: 'Ultimele 7 zile' },
  ];

  const handleAnalyzeText = async () => {
    if (!customText.trim()) return;

    setIsAnalyzing(true);
    try {
      // Mock API call - in real app this would call the sentiment API
      await new Promise(resolve => setTimeout(resolve, 2000));

      setAnalysisResult({
        text: customText,
        sentiment: {
          label: Math.random() > 0.5 ? 'positive' : 'negative',
          score: (Math.random() - 0.5) * 2,
          confidence: Math.random() * 0.3 + 0.7,
        },
        entities: {
          currencies: ['RON', 'EUR'],
          organizations: ['BNR'],
          financial_terms: ['dobândă', 'investiție'],
        },
      });
    } catch (error) {
      console.error('Error analyzing text:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSentimentColor = (label: string) => {
    switch (label) {
      case 'positive': return 'success';
      case 'negative': return 'error';
      default: return 'default';
    }
  };

  const getSentimentLabel = (label: string) => {
    switch (label) {
      case 'positive': return 'Pozitiv';
      case 'negative': return 'Negativ';
      default: return 'Neutru';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analiză de Sentiment
      </Typography>

      <Grid container spacing={3}>
        {/* Currency Selection */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Selecție Monedă
              </Typography>
              <FormControl fullWidth margin="normal">
                <InputLabel>Monedă</InputLabel>
                <Select
                  value={selectedCurrency}
                  label="Monedă"
                  onChange={(e) => setSelectedCurrency(e.target.value)}
                >
                  {currencies.map((currency) => (
                    <MenuItem key={currency.value} value={currency.value}>
                      {currency.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth margin="normal">
                <InputLabel>Perioadă de timp</InputLabel>
                <Select
                  value={timeframe}
                  label="Perioadă de timp"
                  onChange={(e) => setTimeframe(e.target.value)}
                >
                  {timeframes.map((tf) => (
                    <MenuItem key={tf.value} value={tf.value}>
                      {tf.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>

        {/* Custom Text Analysis */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analiză Text Personalizat
              </Typography>

              <TextField
                fullWidth
                multiline
                rows={4}
                placeholder="Introduceți text pentru analiză (știri, tweet-uri, etc.)"
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                margin="normal"
              />

              <Box mt={2} display="flex" justifyContent="flex-end">
                <Button
                  variant="contained"
                  startIcon={isAnalyzing ? <CircularProgress size={20} /> : <SendIcon />}
                  onClick={handleAnalyzeText}
                  disabled={!customText.trim() || isAnalyzing}
                >
                  {isAnalyzing ? 'Se analizează...' : 'Analizează Text'}
                </Button>
              </Box>

              {analysisResult && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Analiza completată cu succes!
                  </Typography>
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Evoluția Sentimentului - {selectedCurrency}
              </Typography>
              <SentimentChart currency={selectedCurrency} timeframe={timeframe} />
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Results */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rezultate Analiză
              </Typography>

              {analysisResult ? (
                <Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Sentiment Detectat
                    </Typography>
                    <Chip
                      label={getSentimentLabel(analysisResult.sentiment.label)}
                      color={getSentimentColor(analysisResult.sentiment.label)}
                      variant="outlined"
                    />
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Scor: {(analysisResult.sentiment.score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Confidență: {(analysisResult.sentiment.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <EntityList entities={analysisResult.entities} />
                </Box>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  Introduceți text pentru analiză pentru a vedea rezultatele aici.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Statistics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Statistici Sentiment - {selectedCurrency}
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="success.main">
                      65%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Sentiment Pozitiv
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="warning.main">
                      25%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Sentiment Neutru
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="error.main">
                      10%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Sentiment Negativ
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="info.main">
                      2.3k
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Puncte de Date
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SentimentAnalysis;
