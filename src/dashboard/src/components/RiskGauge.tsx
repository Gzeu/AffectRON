import React from 'react';
import { Box, Typography } from '@mui/material';

interface RiskGaugeProps {
  score: number; // 0-1 scale
}

const RiskGauge: React.FC<RiskGaugeProps> = ({ score }) => {
  const percentage = score * 100;
  const circumference = 2 * Math.PI * 45; // radius = 45
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const getRiskColor = (score: number) => {
    if (score < 0.3) return '#4caf50'; // Green - Low risk
    if (score < 0.6) return '#ff9800'; // Orange - Medium risk
    if (score < 0.8) return '#f44336'; // Red - High risk
    return '#d32f2f'; // Dark red - Critical risk
  };

  const getRiskLevel = (score: number) => {
    if (score < 0.3) return 'Scăzut';
    if (score < 0.6) return 'Mediu';
    if (score < 0.8) return 'Ridicat';
    return 'Critic';
  };

  const getRiskDescription = (score: number) => {
    if (score < 0.3) return 'Risc minim - condiții favorabile pentru investiții';
    if (score < 0.6) return 'Risc moderat - atenție sporită recomandată';
    if (score < 0.8) return 'Risc ridicat - reconsiderați strategiile de investiții';
    return 'Risc critic - măsuri imediate necesare';
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="center">
      <Box position="relative" width={120} height={120}>
        {/* Background circle */}
        <svg width={120} height={120} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={60}
            cy={60}
            r={45}
            stroke="#e0e0e0"
            strokeWidth={8}
            fill="none"
          />
          {/* Progress circle */}
          <circle
            cx={60}
            cy={60}
            r={45}
            stroke={getRiskColor(score)}
            strokeWidth={8}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            style={{
              transition: 'stroke-dashoffset 0.5s ease-in-out',
            }}
          />
        </svg>

        {/* Center text */}
        <Box
          position="absolute"
          top={0}
          left={0}
          right={0}
          bottom={0}
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
        >
          <Typography variant="h4" fontWeight="bold">
            {percentage.toFixed(0)}%
          </Typography>
          <Typography variant="caption" color="textSecondary">
            {getRiskLevel(score)}
          </Typography>
        </Box>
      </Box>

      <Box mt={2} textAlign="center" maxWidth={200}>
        <Typography variant="body2" color="textSecondary">
          {getRiskDescription(score)}
        </Typography>
      </Box>
    </Box>
  );
};

export default RiskGauge;
