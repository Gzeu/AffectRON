import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Grid,
  Divider,
} from '@mui/material';

interface EntityListProps {
  entities: {
    currencies?: string[];
    organizations?: string[];
    financial_terms?: string[];
  };
}

const EntityList: React.FC<EntityListProps> = ({ entities }) => {
  const renderEntitySection = (title: string, items: string[] = [], color: string) => {
    if (!items || items.length === 0) return null;

    return (
      <Box mb={2}>
        <Typography variant="subtitle2" gutterBottom>
          {title}
        </Typography>
        <Box display="flex" flexWrap="wrap" gap={0.5}>
          {items.map((item, index) => (
            <Chip
              key={index}
              label={item}
              size="small"
              variant="outlined"
              color={color as any}
            />
          ))}
        </Box>
      </Box>
    );
  };

  return (
    <Box>
      {renderEntitySection('Monede', entities.currencies, 'primary')}
      {renderEntitySection('Organizații', entities.organizations, 'secondary')}
      {renderEntitySection('Termeni Financiari', entities.financial_terms, 'success')}

      {(!entities.currencies?.length && !entities.organizations?.length && !entities.financial_terms?.length) && (
        <Typography variant="body2" color="textSecondary">
          Nu au fost detectate entități în text
        </Typography>
      )}
    </Box>
  );
};

export default EntityList;
