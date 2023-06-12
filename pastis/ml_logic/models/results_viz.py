import matplotlib.pyplot as plt
import csv
from params import CSV_PATH

def get_csv_data(file_name):
    # file_name = CSV_PATH
    file_name = 'models_output/csv/0.302_baseline_unet_20230609-144414_csvlog.csv'

    d = {'epoch':[],
        'acc':[],
        'iou':[],
        'loss':[],
        'mean_iou':[],
        'val_acc':[],
        'val_iou':[],
        'val_loss':[],
        'val_mean_iou':[]}

    with open(file_name, 'r') as csvfile:
        data = csv.DictReader(csvfile, skipinitialspace=True)
        for row in data:
            # row is a dict
            d['epoch'].append(float(row['epoch']))
            d['acc'].append(float(row['acc']))
            d['iou'].append(float(row['iou']))
            d['loss'].append(float(row['loss']))
            d['mean_iou'].append(float(row['mean_iou']))
            d['val_acc'].append(float(row['val_acc']))
            d['val_iou'].append(float(row['val_iou']))
            d['val_loss'].append(float(row['val_loss']))
            d['val_mean_iou'].append(float(row['val_mean_iou']))

    return d

def plot_history(history:dict, title='', axs=None, model_name=""):
    if axs is not None:
        ax1, ax2, ax3 = axs
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    if len(model_name) > 0 and model_name[0] != '_':
        model_name = '_' + model_name
    ax1.plot(history['loss'], label = 'train loss' + model_name)
    ax1.plot(history['val_loss'], label = 'val' + model_name)
    ax1.set_ylim(1, 2.75)
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['acc'], label='train accuracy'  + model_name)
    ax2.plot(history['val_acc'], label='val accuracy'  + model_name)
    ax2.set_ylim(0., 1)
    ax2.set_title('Accuracy')
    ax2.legend()

    ax3.plot(history['mean_iou'], label='train mean IuO'  + model_name)
    ax3.plot(history['val_mean_iou'], label='val mean IoU'  + model_name)
    ax3.set_ylim(0., 1)
    ax3.set_title('Mean IuO')
    ax3.legend()

    return (ax1, ax2, ax3)
