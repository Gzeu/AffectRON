import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Chip,
} from '@mui/material';

const Alerts: React.FC = () => {
  const alerts = [
    {
      id: 1,
      type: 'risk',
      severity: 'medium',
      title: 'Volatilitate crescută EUR/RON',
      message: 'Detectată creștere semnificativă a volatilității în perechea EUR/RON',
      timestamp: new Date(),
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Alerte Sistem
      </Typography>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Alerte Active
          </Typography>
          <List>
            {alerts.map((alert) => (
              <ListItem key={alert.id}>
                <ListItemText
                  primary={alert.title}
                  secondary={alert.message}
                />
                <Chip label={alert.severity} color="warning" size="small" />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Alerts;
