import matplotlib.pyplot as plt

def plot_history(history, title='', axs=None, model_name=""):
    if axs is not None:
        ax1, ax2, ax3 = axs
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    if len(model_name) > 0 and model_name[0] != '_':
        model_name = '_' + model_name
    ax1.plot(history.history['loss'], label = 'train loss' + model_name)
    ax1.plot(history.history['val_loss'], label = 'val' + model_name)
    ax1.set_ylim(1, 2.75)
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history.history['acc'], label='train accuracy'  + model_name)
    ax2.plot(history.history['val_acc'], label='val accuracy'  + model_name)
    ax2.set_ylim(0., 1)
    ax2.set_title('Accuracy')
    ax2.legend()

    ax3.plot(history.history['mean_iou'], label='train mean IuO'  + model_name)
    ax3.plot(history.history['val_mean_iou'], label='val mean IoU'  + model_name)
    ax3.set_ylim(0., 1)
    ax3.set_title('Mean IuO')
    ax3.legend()

    return (ax1, ax2, ax3)
