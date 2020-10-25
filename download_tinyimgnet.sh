echo "Downloading Data..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping Data..."
unzip tiny-imagenet-200.zip
echo "Last few steps..."
rm -r ./tiny-imagenet-200/test/*
python3 val_data_format.py
find . -name "*.txt" -delete

