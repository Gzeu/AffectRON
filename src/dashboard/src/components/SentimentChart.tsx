import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface SentimentChartProps {
  currency?: string;
  timeframe?: string;
}

const SentimentChart: React.FC<SentimentChartProps> = ({ currency, timeframe }) => {
  // Mock data - in real app this would come from API
  const data = [
    { time: '00:00', sentiment: 0.2, volume: 45 },
    { time: '04:00', sentiment: 0.1, volume: 32 },
    { time: '08:00', sentiment: 0.3, volume: 67 },
    { time: '12:00', sentiment: 0.4, volume: 89 },
    { time: '16:00', sentiment: 0.2, volume: 56 },
    { time: '20:00', sentiment: 0.3, volume: 78 },
    { time: '24:00', sentiment: 0.25, volume: 43 },
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis domain={[-1, 1]} />
        <Tooltip
          formatter={(value: number) => [
            `${(value * 100).toFixed(1)}%`,
            'Sentiment'
          ]}
        />
        <Line
          type="monotone"
          dataKey="sentiment"
          stroke="#1976d2"
          strokeWidth={2}
          dot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default SentimentChart;
