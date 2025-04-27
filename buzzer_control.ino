const int BUZZER_PIN = 8;  // Broche du buzzer

// Fréquences pour différents types d'alertes
const int ALERT_FREQUENCY = 2000;  // Fréquence pour l'alerte de somnolence
const int BEEP_FREQUENCY = 1000;   // Fréquence pour le bip simple
const int ALARM_FREQUENCY = 1500;  // Fréquence pour l'alarme

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  while (!Serial) {
    ; // Attendre que le port série soit disponible
  }
  Serial.println("Arduino connecté");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case 'T':  // Test de connexion
        Serial.println("Arduino connecté");
        break;
        
      case 'B':  // Bip simple
        tone(BUZZER_PIN, BEEP_FREQUENCY, 500);
        delay(500);
        noTone(BUZZER_PIN);
        Serial.println("Bip simple effectué");
        break;
        
      case 'A':  // Alarme (3 bips)
        for (int i = 0; i < 3; i++) {
          tone(BUZZER_PIN, ALARM_FREQUENCY, 200);
          delay(200);
          noTone(BUZZER_PIN);
          delay(200);
        }
        Serial.println("Alarme effectuée");
        break;

      case 'D':  // Détection de somnolence
        // Son d'alerte plus intense
        for (int i = 0; i < 3; i++) {
          tone(BUZZER_PIN, ALERT_FREQUENCY, 300);
          delay(300);
          noTone(BUZZER_PIN);
          delay(100);
        }
        Serial.println("Alerte somnolence");
        break;
    }
  }
} 