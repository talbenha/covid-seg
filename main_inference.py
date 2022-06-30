from backend.predicter import build_predicter
from data.data_loader import load_data
from utils.utils import preprocess_meta_data



def main():
    # capture the config path from the run arguments
    # then process configuration file

    config = preprocess_meta_data()
    evaluater = build_predicter(config)
    test_folders=config["test_folders"]
    for ct_img,lung_gt,covid_gt in zip(*config["test_folders"]):
        config["test_folders"]=[[ct_img],[lung_gt],[covid_gt]]
        data = load_data(config, train_flag=False) # load the data
        evaluater.save_preds(data)

    config["test_folders"]=test_folders




if __name__ == '__main__':

    main()









