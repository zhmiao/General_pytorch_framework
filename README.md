# General pytorch framework for extension

## For testing training using bee and ants data
### First, unzip data
'''
cd supp/bee_ants
unzip hymenoptera_data.zip
cd ../..
'''

### Second, train the model
'''
python python main.py --config ./configs/beeants_plain_061521.yaml --gpu 0 --session 0
'''

### Third, evaluat the model
'''
python python main.py --config ./configs/beeants_plain_061521.yaml --gpu 0 --session 0 --evaluate val
'''


