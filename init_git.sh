#!/bin/bash
# init_git.sh — À exécuter une seule fois pour initialiser le repo

echo " Initialisation du repo Git..."

git init
git add .
git commit -m "initial project structure

- Structure modulaire : scraping, preprocessing, nlp, knowledge_graph
- .gitignore configuré (données, credentials, venv exclus)
- requirements.txt avec toutes les dépendances
- README.md avec documentation de la pipeline"

echo ""
echo "Repo initialisé !"
echo ""
echo " Prochaines étapes :"
echo "   1. Créer le repo sur GitHub : https://github.com/new"
echo "   2. Lier le repo distant :"
echo "      git remote add origin https://github.com/melamyay/thesis-mental-health-nlp.git"
echo "   3. Pousser :"
echo "      git push -u origin main"
