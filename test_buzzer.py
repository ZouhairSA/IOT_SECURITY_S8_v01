import serial
import time
import serial.tools.list_ports

class ArduinoBuzzer:
    def __init__(self, port=None, baudrate=9600):
        """Initialise la connexion avec l'Arduino."""
        self.port = port
        self.baudrate = baudrate
        self.arduino = None

    def connect(self):
        """Établit la connexion avec l'Arduino."""
        if self.port is None:
            # Essayer de trouver automatiquement le port Arduino
            ports = [port.device for port in serial.tools.list_ports.comports()]
            print("Ports disponibles:", ports)
            
            for port in ports:
                try:
                    print(f"\nEssai de connexion sur {port}...")
                    self.arduino = serial.Serial(port, self.baudrate, timeout=2)
                    time.sleep(2)  # Attendre que la connexion soit établie
                    
                    # Vider le buffer de réception
                    self.arduino.reset_input_buffer()
                    self.arduino.reset_output_buffer()
                    
                    # Essayer plusieurs fois la connexion
                    for _ in range(3):
                        if self.test_connection():
                            self.port = port
                            print(f"✓ Arduino trouvé sur {port}")
                            return True
                        time.sleep(1)
                    
                    self.arduino.close()
                except serial.SerialException as e:
                    print(f"Erreur sur {port}: {e}")
                    continue
            print("\n❌ Aucun Arduino trouvé sur les ports disponibles")
            return False
        else:
            try:
                print(f"\nConnexion sur {self.port}...")
                self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)
                time.sleep(2)
                
                # Vider le buffer de réception
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()
                
                if self.test_connection():
                    return True
                return False
            except serial.SerialException as e:
                print(f"Erreur de connexion à l'Arduino: {e}")
                return False

    def disconnect(self):
        """Ferme la connexion avec l'Arduino."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Déconnecté de l'Arduino")

    def test_connection(self):
        """Teste la connexion avec l'Arduino."""
        try:
            print("Envoi du test de connexion...")
            self.arduino.write(b'T')
            time.sleep(0.5)  # Attendre un peu pour la réponse
            
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                print(f"Réponse reçue: {response}")
                if response == "Arduino connecté":
                    print("✓ Arduino connecté et prêt")
                    return True
            else:
                print("Pas de réponse de l'Arduino")
            return False
        except Exception as e:
            print(f"Erreur lors du test de connexion: {e}")
            return False

    def single_beep(self):
        """Fait un seul bip."""
        if not self.arduino:
            print("❌ Non connecté à l'Arduino")
            return
        try:
            print("\nEnvoi de la commande 'B'...")
            self.arduino.write(b'B')
            time.sleep(0.5)  # Attendre un peu pour la réponse
            
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                print(f"→ {response}")
            else:
                print("Pas de réponse de l'Arduino")
        except Exception as e:
            print(f"Erreur lors de l'envoi de la commande: {e}")

    def alarm(self):
        """Fait une séquence de 3 bips."""
        if not self.arduino:
            print("❌ Non connecté à l'Arduino")
            return
        try:
            print("\nEnvoi de la commande 'A'...")
            self.arduino.write(b'A')
            time.sleep(0.5)  # Attendre un peu pour la réponse
            
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                print(f"→ {response}")
            else:
                print("Pas de réponse de l'Arduino")
        except Exception as e:
            print(f"Erreur lors de l'envoi de la commande: {e}")

def main():
    print("=== Programme de test du buzzer Arduino ===")
    print("Assurez-vous que:")
    print("1. L'Arduino est branché")
    print("2. Le code Arduino est téléchargé")
    print("3. Le buzzer est connecté à la broche 8")
    print("4. Aucun autre programme n'utilise le port série\n")
    
    # Créer une instance de ArduinoBuzzer sans port spécifique
    buzzer = ArduinoBuzzer()
    
    if buzzer.connect():
        try:
            while True:
                print("\nMenu:")
                print("1. Bip simple")
                print("2. Alarme (3 bips)")
                print("3. Quitter")
                
                choix = input("Votre choix (1-3): ")
                
                if choix == '1':
                    buzzer.single_beep()
                elif choix == '2':
                    buzzer.alarm()
                elif choix == '3':
                    break
                else:
                    print("Choix invalide. Veuillez choisir 1, 2 ou 3.")
                    
        except KeyboardInterrupt:
            print("\nArrêt du programme...")
        finally:
            buzzer.disconnect()
    else:
        print("\nImpossible de se connecter à l'Arduino. Vérifiez que:")
        print("1. L'Arduino est bien branché")
        print("2. Le code Arduino est téléchargé")
        print("3. Le port COM est correct")
        print("4. Aucun autre programme n'utilise le port série")

if __name__ == "__main__":
    main()
