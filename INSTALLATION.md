# Guide d'Installation

Cette application fonctionne **avec ou sans GPU**. Choisis l'option qui correspond à ta configuration.

## Option 1 : Installation CPU (tous les ordinateurs)

Fonctionne sur tous les PC, mais plus lent si tu as un GPU.

```powershell
pip install -r requirements-cpu.txt
streamlit run app.py
```

## Option 2 : Installation GPU (avec NVIDIA GPU)

**Prérequis :**
- Carte graphique NVIDIA (GeForce RTX, Tesla, etc.)
- Drivers NVIDIA à jour
- CUDA 11.8 installé sur le système

**Installation :**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
streamlit run app.py
```

Si tu as une version différente de CUDA, remplace `cu118` par ta version (ex: `cu121`, `cu111`).

### Vérifier que le GPU est détecté

```powershell
python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Après installation

Au lancement de l'app Streamlit :
- ✅ Si tu as GPU → checkbox "Utiliser GPU si disponible" cochée par défaut
- ✅ Si tu n'as pas GPU → checkbox décochée, app fonctionne en CPU
- ✅ Tu peux cocher/décocher manuellement dans la barre latérale

### Dépannage spécifique Streamlit Cloud
Si l'app plante à l'import `import cv2` (erreur `ModuleNotFoundError: No module named 'cv2'`), c'est généralement dû à une incompatibilité entre la version Python fournie par l'environnement Streamlit Cloud et les roues OpenCV disponibles.

Solutions recommandées :
- Dans les settings Streamlit Cloud, forcer une version de Python prise en charge (par exemple Python 3.11) si l'option est disponible.
- Pinner une version compatible de `opencv-python-headless` dans `requirements.txt`.
- Si tu contrôles le serveur (ex: AWS), utilise `environment.yml` pour garantir CUDA/Python/apt deps.

Pendant que je détecte automatiquement si `cv2` est présent, l'application affichera maintenant un message clair dans l'UI si OpenCV est absent.

## Dépannage

**Q: J'ai un GPU mais la checkbox dit "GPU non disponible"**
- Vérifiez vos drivers NVIDIA : `nvidia-smi`
- Réinstallez PyTorch CUDA : voir Option 2 ci-dessus

**Q: L'app est lente même avec GPU coché**
- Le modèle peut être limité par d'autres ressources
- Vérifiez que `nvidia-smi` montre l'utilisation du GPU pendant l'inférence

