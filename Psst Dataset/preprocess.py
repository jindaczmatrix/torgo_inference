import os
import re
from datasets import load_dataset

read_directory = "/home/data1/psst-data/psst-data-2022-03-02-full/"
write_directory = "/home/data1"
test_directory = "test"
train_directory = "train"
valid_directory = "valid"
tsv_file_name = "utterances.tsv"
file_name = "utterances.csv"
remove_columns = ["test", "aq_index", "duration_frames"]


def convert_tsv_data_to_csv(dataset):
    """
    Reads in either the psst training, testing or validation data and writes is to a csv file. The function returns the dataset with 
    the desired column information

    Args: dataset - The dataset to read and convert to csv. 
    """
    try:
        
        working_directory = read_directory + dataset
        file_path = os.path.join(write_directory, dataset + "_" + file_name)
        
        if not os.path.isfile(file_path):
            # Read respective tsv file and the write to test_utterances.csv
            with open (working_directory + "/" + tsv_file_name, 'r') as myfile:
                with open(file_path, 'w') as csv_file:
                    for line in myfile:

                        fileContent = re.sub("\t", ",", line)
                        csv_file.write(fileContent)
            
        # modify columns in the data set    
        psstData = load_dataset('csv', data_files= write_directory + "/" + dataset + "_" + file_name)
        psstData = psstData.remove_columns(remove_columns)

        print(psstData)
        return psstData
    
    except FileNotFoundError as e:
        print("Error: ", e)


def main():

    test_data = convert_tsv_data_to_csv(test_directory)
    train_data = convert_tsv_data_to_csv(train_directory)
    valid_data = convert_tsv_data_to_csv(valid_directory)


if __name__=="__main__":
    main()
