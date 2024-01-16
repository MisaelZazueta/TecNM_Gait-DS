import sklearn
from PIL import Image
import os.path
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from metrics import *


def evaluate(model_dir, model_name, data_dir):
    """
    Tests the model by running inference on the test set and evaluating the results.

    :param model_dir: relative path of the trained model folder
    :param model_name: name of the trained model folder
    :param data_dir: relative path to the dataset (folder containing train and test subfolders where each
                     has subfolder which represent one label)
    """
    # load model for inference
    path = model_dir + model_name + "/" + os.listdir(model_dir + model_name)[-1] + "/model/"
    classifier = VisionClassifierInference(
        feature_extractor=ViTFeatureExtractor.from_pretrained(path),
        model=ViTForImageClassification.from_pretrained(path),
    )

    # run inference to get predictions
    pbar = tqdm(desc="Running Inference on Test Set")
    real_labels = []
    pred_labels = []
    for label in os.listdir(data_dir + "/test/"):
        for img in os.listdir(data_dir + "/test/" + label + "/"):
            pbar.update(1)
            image = Image.open(data_dir + "/test/" + label + "/" + img).convert('RGB')
            label_pred = classifier.predict_image(image, False)

    pbar.close()



if __name__ == '__main__':
    evaluate(model_dir="./", model_name="MODEL/DINO-FACEBOOK", data_dir="/users/Maquinot/test//")
