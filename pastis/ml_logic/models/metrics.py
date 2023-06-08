from tensorflow import keras
from keras.metrics import MeanIoU, IoU
import numpy as np

class m_iou():
    def __init__(self, classes: int) -> None:
        self.classes = classes
    def mean_iou(self,y_true, y_pred):
        # y_pred = np.argmax(y_pred, axis = 3)
        # y_true = np.argmax(y_true, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes, sparse_y_pred=False, sparse_y_true=False)
        miou_keras.update_state(y_true, y_pred)
        return miou_keras.result().numpy()
    def miou_class(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes)
        miou_keras.update_state(y_true, y_pred)
        values = np.array(miou_keras.get_weights()).reshape(self.classes, self.classes)
        for i in  range(self.classes):
            class_iou = values[i,i] / (sum(values[i,:]) + sum(values[:,i]) - values[i,i])
            print(f'IoU for class{str(i + 1)} is: {class_iou}')

class _iou():
    def __init__(self, classes: int) -> None:
        self.classes = classes
    def iou(self,y_true, y_pred):
        iou_keras = IoU(num_classes= self.classes, target_class_ids=list(range(0,self.classes)), sparse_y_pred=False, sparse_y_true=False)
        iou_keras.update_state(y_true, y_pred)
        return iou_keras.result().numpy()
    def iou_class(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        iou_keras = IoU(num_classes= self.classes, target_class_ids=list(range(0,self.classes)))
        iou_keras.update_state(y_true, y_pred)
        values = np.array(iou_keras.get_weights()).reshape(self.classes, self.classes)
        for i in  range(self.classes):
            class_iou = values[i,i] / (sum(values[i,:]) + sum(values[:,i]) - values[i,i])
            print(f'IoU for class{str(i + 1)} is: {class_iou}')
