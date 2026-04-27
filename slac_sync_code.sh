rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='results' --exclude=".env" --exclude="plots"  /home/gregor/JAXTPC/ s3df:/sdf/home/g/gregork/jaxtpc
