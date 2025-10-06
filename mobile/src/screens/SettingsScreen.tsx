import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
} from 'react-native';
import {Card, Title, List, Divider} from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialIcons';

const SettingsScreen: React.FC = () => {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [pushNotificationsEnabled, setPushNotificationsEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [vibrationEnabled, setVibrationEnabled] = useState(true);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [darkModeEnabled, setDarkModeEnabled] = useState(false);
  const [language, setLanguage] = useState('ro');

  const languages = [
    {code: 'ro', name: 'Română'},
    {code: 'en', name: 'English'},
  ];

  const refreshIntervals = [
    {value: 15, label: '15 secunde'},
    {value: 30, label: '30 secunde'},
    {value: 60, label: '1 minut'},
    {value: 300, label: '5 minute'},
  ];

  const showAboutAlert = () => {
    Alert.alert(
      'Despre AffectRON',
      'AffectRON Mobile v1.0.0\n\nAnaliză financiară de sentiment pentru piețele românești.\n\nDezvoltat pentru Android și iOS.',
      [{text: 'OK'}],
    );
  };

  const showPrivacyAlert = () => {
    Alert.alert(
      'Politica de Confidențialitate',
      'AffectRON respectă confidențialitatea dumneavoastră. Datele sunt procesate local și nu sunt partajate cu terți fără consimțământ.',
      [{text: 'OK'}],
    );
  };

  const renderNotificationSettings = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title style={styles.sectionTitle}>Notificări</Title>

        <List.Item
          title="Notificări activate"
          description="Primești notificări pentru alerte importante"
          left={props => <List.Icon {...props} icon="bell" />}
          right={props => (
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotificationsEnabled}
            />
          )}
        />

        <Divider style={styles.divider} />

        <List.Item
          title="Notificări push"
          description="Notificări push pe dispozitiv"
          left={props => <List.Icon {...props} icon="cellphone" />}
          right={props => (
            <Switch
              value={pushNotificationsEnabled}
              onValueChange={setPushNotificationsEnabled}
            />
          )}
        />

        <Divider style={styles.divider} />

        <List.Item
          title="Sunet notificări"
          description="Sunet pentru alerte noi"
          left={props => <List.Icon {...props} icon="volume-up" />}
          right={props => (
            <Switch
              value={soundEnabled}
              onValueChange={setSoundEnabled}
            />
          )}
        />

        <Divider style={styles.divider} />

        <List.Item
          title="Vibrație"
          description="Vibrație pentru alerte"
          left={props => <List.Icon {...props} icon="vibration" />}
          right={props => (
            <Switch
              value={vibrationEnabled}
              onValueChange={setVibrationEnabled}
            />
          )}
        />
      </Card.Content>
    </Card>
  );

  const renderDataSettings = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title style={styles.sectionTitle}>Date și Sincronizare</Title>

        <List.Item
          title="Actualizare automată"
          description="Actualizează datele automat în fundal"
          left={props => <List.Icon {...props} icon="sync" />}
          right={props => (
            <Switch
              value={autoRefreshEnabled}
              onValueChange={setAutoRefreshEnabled}
            />
          )}
        />

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="timer" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Interval actualizare</Text>
              <Text style={styles.settingItemDescription}>
                {refreshIntervals.find(interval => interval.value === refreshInterval)?.label || '30 secunde'}
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="storage" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Șterge cache</Text>
              <Text style={styles.settingItemDescription}>
                Eliberează spațiu de stocare
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>
      </Card.Content>
    </Card>
  );

  const renderAppearanceSettings = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title style={styles.sectionTitle}>Aspect și Limbă</Title>

        <List.Item
          title="Mod întunecat"
          description="Schimbă tema aplicației"
          left={props => <List.Icon {...props} icon="brightness-6" />}
          right={props => (
            <Switch
              value={darkModeEnabled}
              onValueChange={setDarkModeEnabled}
            />
          )}
        />

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="language" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Limbă</Text>
              <Text style={styles.settingItemDescription}>
                {languages.find(lang => lang.code === language)?.name || 'Română'}
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>
      </Card.Content>
    </Card>
  );

  const renderAccountSettings = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title style={styles.sectionTitle}>Cont și Securitate</Title>

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="person" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Profil utilizator</Text>
              <Text style={styles.settingItemDescription}>
                Gestionează informațiile contului
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="lock" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Schimbă parolă</Text>
              <Text style={styles.settingItemDescription}>
                Actualizează parola contului
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="backup" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Backup date</Text>
              <Text style={styles.settingItemDescription}>
                Salvează setările și preferințele
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>
      </Card.Content>
    </Card>
  );

  const renderSupportSettings = () => (
    <Card style={styles.card}>
      <Card.Content>
        <Title style={styles.sectionTitle}>Suport și Informații</Title>

        <TouchableOpacity style={styles.settingItem} onPress={showAboutAlert}>
          <View style={styles.settingItemLeft}>
            <Icon name="info" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Despre aplicație</Text>
              <Text style={styles.settingItemDescription}>
                Informații despre AffectRON Mobile
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem} onPress={showPrivacyAlert}>
          <View style={styles.settingItemLeft}>
            <Icon name="security" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Confidențialitate</Text>
              <Text style={styles.settingItemDescription}>
                Politica de confidențialitate
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>

        <Divider style={styles.divider} />

        <TouchableOpacity style={styles.settingItem}>
          <View style={styles.settingItemLeft}>
            <Icon name="help" size={24} color="#666" />
            <View style={styles.settingItemContent}>
              <Text style={styles.settingItemTitle}>Ajutor și suport</Text>
              <Text style={styles.settingItemDescription}>
                Contactează echipa de suport
              </Text>
            </View>
          </View>
          <Icon name="chevron-right" size={24} color="#ccc" />
        </TouchableOpacity>
      </Card.Content>
    </Card>
  );

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Setări</Text>
        <Text style={styles.headerSubtitle}>
          Configurează aplicația după preferințele tale
        </Text>
      </View>

      {/* Notification Settings */}
      {renderNotificationSettings()}

      {/* Data Settings */}
      {renderDataSettings()}

      {/* Appearance Settings */}
      {renderAppearanceSettings()}

      {/* Account Settings */}
      {renderAccountSettings()}

      {/* Support Settings */}
      {renderSupportSettings()}

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          AffectRON Mobile v1.0.0
        </Text>
        <Text style={styles.footerSubtext}>
          © 2024 AffectRON. Toate drepturile rezervate.
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#1976d2',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#e3f2fd',
    marginTop: 4,
  },
  card: {
    margin: 10,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    marginBottom: 10,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
  },
  settingItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingItemContent: {
    marginLeft: 12,
    flex: 1,
  },
  settingItemTitle: {
    fontSize: 16,
    color: '#333',
  },
  settingItemDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  divider: {
    marginVertical: 8,
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  footerSubtext: {
    fontSize: 12,
    color: '#999',
  },
});

export default SettingsScreen;
