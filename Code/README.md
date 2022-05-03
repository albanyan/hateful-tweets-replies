## Installing Requirements
```
pip install -r requirements.txt
```
Note: the code supports running on GPU and on CPU. To run on GPU, install torch version 1.0.1 with the corresponding Cuda version for your device from: https://pytorch.org/get-started/previous-versions/

## Fetch Tweets 
Fatch tweets using the provided tweets IDs for hateful and reply tweets in " \data\hateful_tweets-replies.csv ". Then, save the output file in the same folder " \data\ " using the same file's name " hateful_tweets-replies.csv ".

## Preparing the Dataset
To prepare the dataset for training, run the following command:
```
python prepare_data.py --csv-file {file_path} --question {question_number} --model-name {model_name} --output-dir {output_path}
```
Where:  
`file_path`: path of the .csv data file.  
`question_number`: question (task) number ranging from 1 to 4.  
`model_name`: either "**bert**" or "**bertweet**".  
`output_path`: directory to save the processed data (default is ./Dataloaders/).  

## Training
```
python train.py --data-dir {processed_data_path} --output-dir {output_path} --model-name {model_name}
```
Where:  
`processed_data_path`: directory of the processed data (default is ./Dataloaders/).  
`output_path`: directory to save the trained model (default is ./Output/).  
`model_name`: either "**bert**" or "**bertweet**".

## Testing
```
python test.py --data-dir {processed_data_path} --trained-model-dir {trained_model_path} --output-dir {output_path}
```
Where:  
`processed_data_path`: directory of the processed data (default is ./Dataloaders/).  
`trained_model_path`: directory of the trained model (default ./Output/).  
`output_path`: directory to save the predictions (default is ./Output/).  
