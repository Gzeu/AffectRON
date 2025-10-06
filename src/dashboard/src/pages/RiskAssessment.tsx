import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
} from '@mui/material';

const RiskAssessment: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Evaluare Riscuri
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analiza Riscurilor de Portofoliu
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Sistemul de evaluare a riscurilor va fi implementat în această secțiune.
                Va include analiza volatilității, corelații și factori de risc specifici.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskAssessment;
