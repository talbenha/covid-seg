from models.wnet_vgg import Wnet_vgg

def build_model(config):
    if config['model_name'] == "HU-Net":
        model = Wnet_vgg(input_size=config["model_input_dim"]).build()

    else:
        raise ValueError("'{}' is an invalid model name")

    return model


