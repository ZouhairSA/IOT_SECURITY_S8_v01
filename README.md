# IOT_SECURITY_S8_v01

## 🛡️ Système de Détection de Somnolence avec Alerte Buzzer (Arduino + Python + IA)

Ce projet propose un système intelligent de détection de somnolence en temps réel, utilisant la vision par ordinateur (YOLO, MediaPipe), une interface graphique moderne (PyQt5), et un buzzer connecté à un Arduino pour l'alerte physique.

---

## 🚀 Fonctionnalités principales
- **Détection de clignements, micro-sommeil et bâillements** via webcam (USB recommandée)
- **Interface graphique professionnelle** (PyQt5) avec sélection de caméra et port Arduino
- **Affichage en temps réel** des points clés sur les yeux et la bouche
- **Alerte sonore** : buzzer Arduino déclenché lors d'une détection de somnolence
- **Statistiques** : nombre de clignements, bâillements, durée de micro-sommeil, etc.
- **Compatible Windows** (Python 3.10+ recommandé)

---

## 🛠️ Matériel nécessaire
- 1 x Arduino Uno/Nano/compatible
- 1 x Buzzer passif
- 1 x Webcam USB
- 1 x Câble USB pour Arduino
- Fils de connexion

### Schéma de branchement
- **Buzzer** :
  - Broche + du buzzer → Pin 8 de l'Arduino
  - Broche - du buzzer → GND de l'Arduino
- **USB** :
  - Arduino relié au PC via USB
  - Webcam USB branchée au PC

---

## 💻 Installation logicielle
1. **Cloner le dépôt**
   ```sh
   git clone https://github.com/ZouhairSA/IOT_SECURITY_S8_v01.git
   cd IOT_SECURITY_S8_v01
   ```
2. **Installer les dépendances Python**
   ```sh
   pip install -r requirements.txt
   ```
   (PyQt5, opencv-python, numpy, ultralytics, mediapipe, pyserial, etc.)
3. **Téléverser le code Arduino**
   - Ouvrir `buzzer_control.ino` dans l'IDE Arduino
   - Sélectionner la bonne carte et le bon port
   - Télécharger sur l'Arduino

---

## 🖥️ Utilisation
1. **Branchez la webcam USB et l'Arduino**
2. **Lancez l'application Python**
   ```sh
   python DrowsinessDetector.py
   ```
3. **Sélectionnez la caméra USB** dans l'interface
4. **Sélectionnez le port Arduino** (ex : COM3)
5. **Testez** : fermez les yeux ou baillez pour déclencher l'alerte (le buzzer sonne)

---

## 📦 Structure du projet
- `DrowsinessDetector.py` : Interface graphique et détection IA
- `buzzer_control.ino` : Code Arduino pour le buzzer
- `test_buzzer.py` : Script de test du buzzer
- `requirements.txt` : Dépendances Python
- `README.md` : Ce fichier

---

## 🤝 Contribution
Les contributions sont les bienvenues !
- Forkez le repo
- Créez une branche (`feature/ma-fonctionnalite`)
- Faites un commit
- Ouvrez une Pull Request

---

## 📧 Contact
Pour toute question ou suggestion : ZouhairSA sur GitHub

---

## 📝 Licence
Ce projet est open-source sous licence MIT.

