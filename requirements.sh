#!/bin/sh
print('***************************Python is going to install its requirements*********************************')
pip install  pandas 
pip install  numpy 
pip install  sklearn
pip install matplotlib
pip install  seaborn
pip install  imblearn
pip install  yellowbrick.classifier
pip install  keras

print('************************************************************')

chmod +x model.py
python model.py

model.py
