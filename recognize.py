from joblib import load
import torch
from cnn.cnn import Net
import numpy as np
from future.future_extraction import get_yi_record
from sklearn.metrics import precision_recall_curve
import argparse
import os
import torchaudio
from torchaudio import transforms

torchaudio.backend.set_audio_backend(backend='soundfile')

MODE = 'DEBUG' #'RELEASE'
THRESHOLD = 0.545 #0.95
SAMPLE_RATE = 16000


def get_feature_vector(audio_path: str, model, device):
    # feature_vector = np.empty((1, 400))
    audio = torchaudio.load(audio_path)
    sample_rate = audio[1]
    audio = audio[0]
    audio = transforms.Resample(sample_rate, SAMPLE_RATE)(audio)
    feature_vector = get_yi_record(model, device, audio)
    return feature_vector.reshape((1, -1))

class RecognizeSpeaker:
    def __init__(self):
        with torch.no_grad():
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.cnn = Net().to(self.device)
            self.cnn.load_state_dict(torch.load('data/ResNet_model/model_weights16.pth',
                                               map_location=lambda storage, loc: storage))
            self.cnn.eval()
            # self.cnn = self.cnn.double()

        self.svm_model = load('data/svm_classifier/samy_set_v3_SVM_rbf_C_1_gamma_0.01.pkl')

    def detect(self, audio_path):
        audio_feature_vector = get_feature_vector(audio_path, self.cnn, self.device)
        preds = self.svm_model.predict_proba(audio_feature_vector)[0]
        pred_class = np.argmax(preds)
        return pred_class, preds[pred_class]


def detect(image_path):
    td = RecognizeSpeaker()
    pred_class, score = td.detect(image_path)
    list_classes = os.listdir(r'F:\gavno')
    return list_classes[pred_class], score



if __name__=='__main__':
    if MODE == 'DEBUG':
        #calculate_threshold()
        images_path_dir = 'F:\\test'
        for image_name in os.listdir(images_path_dir):
            image_path = os.path.join(images_path_dir, image_name)
            pred_class, score = detect(image_path)
            print(f'Audio_path: {image_path}, predicted_class - {pred_class}, score - {score}')

