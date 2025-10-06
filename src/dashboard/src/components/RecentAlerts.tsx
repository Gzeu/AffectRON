import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  Typography,
  Chip,
  Box,
  Divider,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

const RecentAlerts: React.FC = () => {
  // Mock data - in real app this would come from API
  const alerts = [
    {
      id: 1,
      type: 'risk',
      severity: 'medium',
      title: 'Volatilitate crescută EUR/RON',
      message: 'Detectată creștere semnificativă a volatilității în perechea EUR/RON',
      timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
    },
    {
      id: 2,
      type: 'sentiment',
      severity: 'high',
      title: 'Sentiment negativ în creștere',
      message: 'Analiza sentimentului arată tendință negativă în ultimele 2 ore',
      timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
    },
    {
      id: 3,
      type: 'system',
      severity: 'low',
      title: 'Actualizare sursă de date',
      message: 'Sursa de știri financiare actualizată cu succes',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    },
  ];

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high':
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'medium':
        return <WarningIcon color="warning" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
      case 'critical':
        return 'error';
      case 'medium':
        return 'warning';
      default:
        return 'info';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diffMinutes = Math.floor((now.getTime() - timestamp.getTime()) / (1000 * 60));

    if (diffMinutes < 60) {
      return `acum ${diffMinutes} minute${diffMinutes !== 1 ? 'e' : ''}`;
    } else {
      const diffHours = Math.floor(diffMinutes / 60);
      return `acum ${diffHours} oră${diffHours !== 1 ? 'e' : ''}`;
    }
  };

  return (
    <List dense>
      {alerts.map((alert, index) => (
        <React.Fragment key={alert.id}>
          <ListItem alignItems="flex-start">
            <Box mr={1} mt={0.5}>
              {getSeverityIcon(alert.severity)}
            </Box>

            <ListItemText
              primary={
                <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                  <Typography variant="subtitle2">
                    {alert.title}
                  </Typography>
                  <Chip
                    label={alert.severity}
                    size="small"
                    color={getSeverityColor(alert.severity)}
                    variant="outlined"
                  />
                </Box>
              }
              secondary={
                <Box>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    {alert.message}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {formatTimestamp(alert.timestamp)}
                  </Typography>
                </Box>
              }
            />
          </ListItem>
          {index < alerts.length - 1 && <Divider />}
        </React.Fragment>
      ))}

      {alerts.length === 0 && (
        <Box textAlign="center" py={2}>
          <Typography variant="body2" color="textSecondary">
            Nu există alerte recente
          </Typography>
        </Box>
      )}
    </List>
  );
};

export default RecentAlerts;
