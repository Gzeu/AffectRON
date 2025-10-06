import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  TextField,
  Button,
} from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Setări Sistem
      </Typography>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Setări Generale
          </Typography>

          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Activare analiză în timp real"
          />

          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Activare notificări push"
          />

          <FormControlLabel
            control={<Switch />}
            label="Mod întunecat"
          />
        </CardContent>
      </Card>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configurare API
          </Typography>

          <TextField
            fullWidth
            label="URL API"
            defaultValue="http://localhost:8000"
            margin="normal"
          />

          <TextField
            fullWidth
            label="Cheie API"
            type="password"
            margin="normal"
          />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Setări Avansate
          </Typography>

          <TextField
            fullWidth
            label="Interval actualizare (secunde)"
            type="number"
            defaultValue="300"
            margin="normal"
          />

          <Button variant="contained" sx={{ mt: 2 }}>
            Salvează Setările
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Settings;
