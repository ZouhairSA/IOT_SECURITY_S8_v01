const int BUZZER_PIN = 9;  // Buzzer connected to pin 9

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
  digitalWrite(BUZZER_PIN, LOW);  // Ensure buzzer is off at start
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') {
      digitalWrite(BUZZER_PIN, HIGH);  // Turn buzzer on
    } else if (command == '0') {
      digitalWrite(BUZZER_PIN, LOW);   // Turn buzzer off
    }
  }
}
