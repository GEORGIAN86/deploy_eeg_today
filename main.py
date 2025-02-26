from preprocess_filter import preprocess
from model import modal

if __name__ == "__main__":
    data_list, label_list = preprocess()
    modal(data_list, label_list)
