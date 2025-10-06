import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Box,
  Chip,
} from '@mui/material';
import { TrendingUp as TrendingUpIcon, TrendingDown as TrendingDownIcon } from '@mui/icons-material';

const MarketOverview: React.FC = () => {
  // Mock data - in real app this would come from API
  const marketData = [
    {
      pair: 'EUR/RON',
      rate: 4.9750,
      change: 0.25,
      volume: '1.2M',
      source: 'BNR',
    },
    {
      pair: 'USD/RON',
      rate: 4.5800,
      change: -0.15,
      volume: '890K',
      source: 'ECB',
    },
    {
      pair: 'BTC/RON',
      rate: 198500,
      change: 2.8,
      volume: '45K',
      source: 'CryptoCompare',
    },
    {
      pair: 'GBP/RON',
      rate: 5.8200,
      change: 0.05,
      volume: '234K',
      source: 'ECB',
    },
  ];

  return (
    <Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Pereche</TableCell>
              <TableCell align="right">Curs</TableCell>
              <TableCell align="right">Schimbare</TableCell>
              <TableCell align="right">Volum</TableCell>
              <TableCell>Sursă</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {marketData.map((row) => (
              <TableRow key={row.pair}>
                <TableCell component="th" scope="row">
                  <Typography variant="body2" fontWeight="medium">
                    {row.pair}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2">
                    {row.rate.toLocaleString('ro-RO', {
                      minimumFractionDigits: row.pair.includes('BTC') ? 0 : 4,
                      maximumFractionDigits: row.pair.includes('BTC') ? 0 : 4,
                    })}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Box display="flex" alignItems="center" justifyContent="flex-end">
                    {row.change > 0 ? (
                      <TrendingUpIcon color="success" sx={{ fontSize: 16, mr: 0.5 }} />
                    ) : (
                      <TrendingDownIcon color="error" sx={{ fontSize: 16, mr: 0.5 }} />
                    )}
                    <Typography
                      variant="body2"
                      color={row.change > 0 ? 'success.main' : 'error.main'}
                    >
                      {row.change > 0 ? '+' : ''}
                      {row.change.toFixed(2)}%
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2">
                    {row.volume}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip label={row.source} size="small" variant="outlined" />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
        Ultima actualizare: {new Date().toLocaleTimeString('ro-RO')}
      </Typography>
    </Box>
  );
};

export default MarketOverview;
