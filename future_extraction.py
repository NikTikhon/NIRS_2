from cnn.cnn import Net
import torch
from future.future_extraction import create_feature_vectors

model_weight_path = 'data/ResNet_model/69_400d.pth'

with torch.no_grad():
    set_path = r'F:\Ucheba\NIRS\DataSet\train_svm_dataset\\'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_weight_path,
                                     map_location=lambda storage, loc: storage))
    model.eval()
    output_filename = 'data/features/samy_set_v3_WithRot_LR0001_b128.csv'
    create_feature_vectors(model, device, set_path, output_filename, 400)
