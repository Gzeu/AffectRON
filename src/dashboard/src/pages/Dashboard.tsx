import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

import SentimentChart from './SentimentChart';
import MarketOverview from './MarketOverview';
import RiskGauge from './RiskGauge';
import RecentAlerts from './RecentAlerts';

const Dashboard: React.FC = () => {
  // Mock data - in real app this would come from API
  const dashboardData = {
    overall_sentiment: {
      score: 0.2,
      trend: 'up',
      confidence: 0.85,
    },
    market_health: {
      score: 0.7,
      status: 'good',
    },
    risk_level: {
      score: 0.3,
      level: 'low',
    },
    active_alerts: 2,
    data_sources: {
      news: { status: 'active', last_update: '2 min ago' },
      twitter: { status: 'active', last_update: '5 min ago' },
      fx: { status: 'active', last_update: '1 min ago' },
    },
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard Financiar
      </Typography>

      <Grid container spacing={3}>
        {/* Overall Sentiment Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Sentiment General
                  </Typography>
                  <Typography variant="h4">
                    {dashboardData.overall_sentiment.score > 0 ? '+' : ''}
                    {(dashboardData.overall_sentiment.score * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Box>
                  {dashboardData.overall_sentiment.trend === 'up' ? (
                    <TrendingUpIcon color="success" sx={{ fontSize: 40 }} />
                  ) : (
                    <TrendingDownIcon color="error" sx={{ fontSize: 40 }} />
                  )}
                </Box>
              </Box>
              <Box mt={2}>
                <Typography variant="body2" color="textSecondary">
                  Confidență: {(dashboardData.overall_sentiment.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Market Health Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Starea Pieței
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    Bună
                  </Typography>
                </Box>
                <CheckCircleIcon color="success" sx={{ fontSize: 40 }} />
              </Box>
              <Box mt={2}>
                <LinearProgress
                  variant="determinate"
                  value={dashboardData.market_health.score * 100}
                  color="success"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Level Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Nivel de Risc
                  </Typography>
                  <Typography variant="h4" color="warning.main">
                    Scăzut
                  </Typography>
                </Box>
                <WarningIcon color="warning" sx={{ fontSize: 40 }} />
              </Box>
              <Box mt={2}>
                <Chip
                  label={`Scor: ${(dashboardData.risk_level.score * 100).toFixed(1)}%`}
                  color="warning"
                  variant="outlined"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Active Alerts Card */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="h6">
                    Alerte Active
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {dashboardData.active_alerts}
                  </Typography>
                </Box>
                <WarningIcon color="error" sx={{ fontSize: 40 }} />
              </Box>
              <Box mt={2}>
                <Typography variant="body2" color="textSecondary">
                  Necesită atenție
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Trend Chart */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Evoluția Sentimentului (24h)
            </Typography>
            <SentimentChart />
          </Paper>
        </Grid>

        {/* Risk Gauge */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Evaluarea Riscurilor
            </Typography>
            <RiskGauge score={dashboardData.risk_level.score} />
          </Paper>
        </Grid>

        {/* Market Overview */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Prezentare Generală Piață
            </Typography>
            <MarketOverview />
          </Paper>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Alerte Recente
            </Typography>
            <RecentAlerts />
          </Paper>
        </Grid>

        {/* Data Sources Status */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Status Surse de Date
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(dashboardData.data_sources).map(([source, info]) => (
                <Grid item xs={12} sm={6} md={4} key={source}>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Typography variant="body2" textTransform="capitalize">
                      {source === 'fx' ? 'Schimb Valutar' : source === 'news' ? 'Știri' : 'Twitter'}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        label={info.status}
                        color={info.status === 'active' ? 'success' : 'error'}
                        size="small"
                      />
                      <Typography variant="caption" color="textSecondary">
                        {info.last_update}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
