import math

import numpy as np
import torch
import torchaudio
from torchaudio import transforms
import pandas as pd
import os
import glob
from future.future_vision import get_yi, get_y_hat

torchaudio.backend.set_audio_backend(backend='soundfile')

SAMPLE_RATE = 16000

def get_audios_and_labels(audios_path):
    class_map = os.listdir(audios_path)
    file_list = glob.glob(audios_path + "*")

    audios = {}
    for class_path in file_list:
        class_name = class_path.split("\\")[-1]
        for aud_path in glob.glob(class_path + "/*.wav"):
            audios[aud_path] = {}
            audios[aud_path]['aud'] = torchaudio.load(aud_path)
            audios[aud_path]['label'] = class_map.index(class_name)
    return audios


def create_feature_vectors(model, device, data_path, output_name, v_len):
    df = pd.DataFrame()
    audios = get_audios_and_labels(data_path)
    c = 1
    for audio_name in audios.keys():
        print("Audio: ", c)

        audio = audios[audio_name]['aud']
        label = audios[audio_name]['label']
        sample_rate = audio[1]
        audio = audio[0]
        audio = transforms.Resample(sample_rate, SAMPLE_RATE)(audio)

        futures = get_yi_record(model, device, audio)

        df = pd.concat([df, pd.concat([pd.DataFrame([audio_name.split(os.sep)[-1], str(label)]),
                                       pd.DataFrame(futures)])], axis=1, sort=False)

        c += 1

    # save the feature vector to csv
    final_df = df.T
    final_df.columns = get_df_column_names(v_len)
    final_df.to_csv(output_name, index=False)  # csv type [im_name][label][f1,f2,...,fK]


def get_yi_record(model, device, audio):
    sample_rate_record = 16384
    list_slice = [audio[:, i*sample_rate_record: i*sample_rate_record + sample_rate_record] for i in range(math.ceil(audio.size()[1] / sample_rate_record))]
    future_list = []
    for part in list_slice:
        audio_part = torch.fft.fft(part, 16000)[0].reshape((1, 16000))
        audio_part = torch.abs(audio_part)
        audio_part = audio_part[:, :8000]
        audio_part = audio_part.unsqueeze_(0)
        futures = get_yi(model, audio_part, device)
        future_list.append(futures.cpu().numpy())
    future_list = np.array(future_list)
    mean_future = get_y_hat(future_list, 'mean')
    return mean_future.squeeze()




def get_df_column_names(v_len):
    """
    Rename the feature csv column names as [im_names][labels][f1,f2,...,fK].
    :returns: The column names
    """
    names = ["audio_names", "labels"]
    for i in range(v_len):
        names.append("f" + str(i + 1))
    return names