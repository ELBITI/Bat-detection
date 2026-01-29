# Installation et lancement de l'app YOLOv8-Streamlit

## Problème avec Python 3.13

**Ultralytics (YOLOv8)** ne publie pas encore de paquet compatible avec **Python 3.13** sur PyPI.  
Si vous avez uniquement Python 3.13, l'installation de `ultralytics` échouera.

### Solution recommandée : utiliser Python 3.10, 3.11 ou 3.12

1. **Télécharger Python 3.12** (ou 3.11) depuis : https://www.python.org/downloads/
2. **Créer un environnement virtuel** dans le dossier du projet :
   ```powershell
   cd C:\Users\Etudiant\YOLOv8-streamlit-app
   C:\Chemin\Vers\Python312\python.exe -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. **Installer les dépendances** :
   ```powershell
   pip install -r requirements-minimal.txt
   ```
   Ou les paquets essentiels uniquement :
   ```powershell
   pip install streamlit ultralytics opencv-python-headless easyocr Pillow pytube numpy
   ```
4. **Lancer l'application** :
   ```powershell
   streamlit run app.py
   ```
5. Ouvrir dans le navigateur l’URL affichée (souvent **http://localhost:8501**).

---

## Si vous avez déjà Python 3.10, 3.11 ou 3.12

Dans un terminal (PowerShell ou CMD), dans le dossier du projet :

```powershell
cd C:\Users\Etudiant\YOLOv8-streamlit-app
pip install -r requirements-minimal.txt
streamlit run app.py
```

Ou exécuter le script : **`install_et_lancer.ps1`** (clic droit → Exécuter avec PowerShell).

---

## Fichiers utiles

- **requirements-minimal.txt** : dépendances minimales avec versions flexibles
- **requirements.txt** : dépendances complètes avec versions figées (peut être incompatible avec certaines versions de Python)
- **install_et_lancer.ps1** : script qui installe puis lance l’app (vérifie la version de Python)
