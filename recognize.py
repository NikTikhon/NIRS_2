import matplotlib.pyplot as plt
from joblib import load
import torch
from cnn.cnn import Net
import numpy as np
from future.future_extraction import get_yi_record
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score
from visual.visual import plot_conf_m
import os
import torchaudio
from torchaudio import transforms

torchaudio.backend.set_audio_backend(backend='soundfile')
model_weight_path = 'data/ResNet_model/69_400d.pth'

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
            self.cnn.load_state_dict(torch.load(model_weight_path,
                                               map_location=lambda storage, loc: storage))
            self.cnn.eval()

        self.svm_model = load('data/svm_classifier/samy_set_v3_SVM_rbf_C_1_gamma_0.01.pkl')

    def detect(self, audio_path):
        audio_feature_vector = get_feature_vector(audio_path, self.cnn, self.device)
        preds = self.svm_model.predict_proba(audio_feature_vector)[0]
        pred_class = np.argmax(preds)
        # return pred_class, preds[pred_class]
        return pred_class, preds[pred_class]


def detect(audio_path):
    td = RecognizeSpeaker()
    pred_class, score = td.detect(audio_path)
    list_classes = os.listdir(r'F:\Ucheba\NIRS\DataSet\train_svm_dataset')
    # return list_classes[pred_class], score
    return pred_class, score



if __name__=='__main__':
    if MODE == 'DEBUG':
        #calculate_threshold()
        # audio_path_dir = r'F:\Ucheba\NIRS\DataSet\test_svm_dataset\p276'
        # for audio_name in os.listdir(audio_path_dir):
        #     audio_path = os.path.join(audio_path_dir, audio_name)
        #     pred_class, score = detect(audio_path)
        #     print(f'Audio_path: {audio_path}, predicted_class - {pred_class}, score - {score}')
        y_test = []
        y_pred = []
        audio_path_dir = r'F:\Ucheba\NIRS\DataSet\test_svm_dataset\\'
        speakers = os.listdir(audio_path_dir)
        for ind in range(len(speakers)):
            speaker_path = os.path.join(audio_path_dir, speakers[ind])
            audios = os.listdir(speaker_path)
            for audio in audios:
                audio_path = os.path.join(speaker_path, audio)
                pred_class, score = detect(audio_path)
                print(f'Audio_path: {audio_path}, predicted_class - {pred_class}, score - {score}')
                y_test.append(ind)
                y_pred.append(pred_class)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plot_conf_m(conf_matrix, '20 дикторов')
        accuracy = accuracy_score(y_test, y_pred)
        print('----------------------')
        print(conf_matrix)
        print('----------------------')
        print(accuracy)
