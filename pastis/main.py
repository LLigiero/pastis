from ml_logic.baseline_model import baseline_unet_model, compile_model, train_model
from sklearn.model_selection import train_test_split

# Instantiate class instance
pastis = PastisDataset()

# ----- TRAIN TEST SPLIT -----
'''Assuming data already preprocessed'''
# '''Create ds_train, ds_val, ds_test)'''

# ----- TRAIN MODEL -----
'''Define model using `baseline_model.py`'''
model = baseline_unet_model(dropout=0.2)

'''Compile model using `baseline_model.py`'''
model = compile_model(model, learning_rate=0.05)

'''Train model using `baseline_model.py`'''
batch_size = 28
patience = 2

model, history = train_model(
        model,
        train_ds,
        batch_size=batch_size,
        patience=patience,
    )

# ----- EVALUATE MODEL -----
