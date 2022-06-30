from trainers.trainer import build_trainer
from backend.predicter import build_predicter
from data.data_loader import load_data
from models.models import build_model
from utils.utils import preprocess_meta_data
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

def main():
    # capture the config path from the run arguments
    # then process configuration file

    config = preprocess_meta_data()

    # load the data
    data = load_data(config)

    if not config['quiet']:
        print(config)

    # create a model
    model = build_model(config)

    # create trainer
    trainer = build_trainer(model, data, config)

    # train the model
    trainer.train()




if __name__ == '__main__':
    main()







