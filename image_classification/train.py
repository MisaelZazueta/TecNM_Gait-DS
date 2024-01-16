import shutil
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
import os.path
from transformers import ViTFeatureExtractor, ViTForImageClassification


def train_model(model_type, epochs, batch_size, test_ratio, data_dir, output_dir, output_name):
    """
    Runs the fine-tuning of a Vision Transformer model on a prepared dataset.

    :param model_type: official huggingface model_id of the pretrained model to use
    :param epochs: number of epochs to run the training for
    :param batch_size: batch size to use for training
    :param test_ratio: relative size of test set for direct evaluation on test split from augmented train
                       set in HugsVision (between 0 and 1), INDEPENDENT OF THE SELF-IMPLEMENTED MODEL EVALUATION!
    :param data_dir: relative path to the dataset (folder containing train and test subfolders where each
                     has subfolder which represent one label)
    :param output_dir: relative path for the trained model folder to create
    :param output_name: name of the model folder to create
    """
    # load dataset from folder (augmentation and balancing is done externally so deactivated here)
    train, test, id2label, label2id = VisionDataset.fromImageFolder(
        data_dir, test_ratio=test_ratio, balanced=False, augmentation=False)

    # train
    trainer = VisionClassifierTrainer(
        model_name=output_name, train=train, test=test, output_dir=output_dir,
        max_epochs=epochs, batch_size=batch_size, model=ViTForImageClassification.from_pretrained(
            model_type, num_labels=len(label2id), label2id=label2id, id2label=id2label),
        feature_extractor=ViTFeatureExtractor.from_pretrained(model_type),
    )

    # evaluate (direct evaluation on test_split from augmented train set in HugsVision)
    trainer.evaluate_f1_score()

    # copy /feature_extractor/preprocessor_config.json to /model/ to make the model folder loadable for inference
    path = output_dir + output_name + "/" + os.listdir(output_dir + output_name)[-1] + "/"
    shutil.copy(path + "feature_extractor/preprocessor_config.json", path + "model/preprocessor_config.json")


if __name__ == '__main__':
    train_model(model_type="facebook/dino-vitb16", epochs=20, batch_size=16, test_ratio=0.2,
                data_dir=r"C:\Users\Maquinot\train\\", output_dir="./", output_name="model/dino-facebook")
