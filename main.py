
from preprocessing import preprocess
from model import modal
if __name__ == "main":
    data_list,label_list = preprocess()
    modal(data_list,label_list)
    
    