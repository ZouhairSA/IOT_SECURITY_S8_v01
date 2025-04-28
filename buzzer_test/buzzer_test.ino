const int BUZZER_PIN = 9;  // Changez ce numéro selon votre branchement

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    switch(cmd) {
      case 'B':  // Bip simple
        digitalWrite(BUZZER_PIN, HIGH);
        delay(200);
        digitalWrite(BUZZER_PIN, LOW);
        Serial.println("Bip effectué");
        break;
        
      case 'A':  // Alarme (3 bips)
        for(int i = 0; i < 3; i++) {
          digitalWrite(BUZZER_PIN, HIGH);
          delay(200);
          digitalWrite(BUZZER_PIN, LOW);
          delay(200);
        }
        Serial.println("Alarme effectuée");
        break;
        
      case 'T':  // Test de connexion
        Serial.println("Arduino connecté");
        break;
    }
  }
}
