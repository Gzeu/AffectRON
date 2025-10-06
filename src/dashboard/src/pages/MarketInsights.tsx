import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
} from '@mui/material';

const MarketInsights: React.FC = () => {
  // Mock data - in real app this would come from API
  const insights = [
    {
      currency: 'EUR',
      sentiment: 'positive',
      trend: 'up',
      confidence: 0.85,
      recommendations: [
        'Considerați creșterea expunerii pe EUR',
        'Monitorizați volatilitatea pentru oportunități de trading',
      ],
    },
    {
      currency: 'USD',
      sentiment: 'neutral',
      trend: 'sideways',
      confidence: 0.72,
      recommendations: [
        'Mențineți pozițiile actuale',
        'Așteptați semnale mai clare înainte de acțiune',
      ],
    },
    {
      currency: 'RON',
      sentiment: 'positive',
      trend: 'up',
      confidence: 0.91,
      recommendations: [
        'Stabilitate monetară bună',
        'Risc valutar scăzut pentru RON',
      ],
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Insights Piață
      </Typography>

      <Grid container spacing={3}>
        {insights.map((insight) => (
          <Grid item xs={12} md={6} lg={4} key={insight.currency}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">
                    {insight.currency}/RON
                  </Typography>
                  <Chip
                    label={insight.sentiment}
                    color={insight.sentiment === 'positive' ? 'success' : 'default'}
                    size="small"
                  />
                </Box>

                <Box mb={2}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Trend: {insight.trend}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={insight.confidence * 100}
                    color="info"
                  />
                  <Typography variant="caption" color="textSecondary">
                    Confidență: {(insight.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <Typography variant="subtitle2" gutterBottom>
                  Recomandări:
                </Typography>
                <Box>
                  {insight.recommendations.map((rec, index) => (
                    <Typography key={index} variant="body2" gutterBottom>
                      • {rec}
                    </Typography>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default MarketInsights;
