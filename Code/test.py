import argparse
import os
import torch
import numpy as np
import time
import datetime
from sklearn.metrics import f1_score, recall_score, accuracy_score,precision_score
from transformers import AutoModelForSequenceClassification

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def calculate_score(preds, labels):
    y =  np.array(preds)
    y_test = labels
    results = []

    results.append(round(precision_score(y_test,y,average=None)[0],2))
    results.append(round(recall_score(y_test,y,average=None)[0],2))
    results.append(round(f1_score(y_test,y,average=None)[0],2))

    results.append(round(precision_score(y_test,y,average=None)[1],2))
    results.append(round(recall_score(y_test,y,average=None)[1],2))
    results.append(round(f1_score(y_test,y,average=None)[1],2))

    results.append(round(f1_score(y_test,y,average='weighted'),2))

    return results

def test(model, test_dataloader):
	model.eval()
	model.to(DEVICE)
	preds = []
	# Tracking variables 
	total_eval_accuracy = 0
	total_eval_loss = 0
	nb_eval_steps = 0
	t0 = time.time()

	# Evaluate data for one epoch
	for batch in test_dataloader:
			
			# Unpack this training batch from our dataloader. 
			#
			# As we unpack the batch, we'll also copy each tensor to the GPU using 
			# the `to` method.
			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: attention masks
			#   [2]: labels 
			b_input_ids = batch[0].to(DEVICE)
			b_input_mask = batch[1].to(DEVICE)
			b_labels = batch[2].to(DEVICE)
			
			# Tell pytorch not to bother with constructing the compute graph during
			# the forward pass, since this is only needed for backprop (training).
			with torch.no_grad():        

					# Forward pass, calculate logit predictions.
					# token_type_ids is the same as the "segment ids", which 
					# differentiates sentence 1 and 2 in 2-sentence tasks.
					# The documentation for this `model` function is here: 
					# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
					# Get the "logits" output by the model. The "logits" are the output
					# values prior to applying an activation function like the softmax.
					outputs = model(b_input_ids, 
																	token_type_ids=None, 
																	attention_mask=b_input_mask,
																	labels=b_labels)
					loss = outputs.loss
					logits = outputs.logits
					
					
			# Accumulate the validation loss.
			total_eval_loss += loss.item()

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			preds.append(logits)
			# Calculate the accuracy for this batch of test sentences, and
			# accumulate it over all batches.
			total_eval_accuracy += flat_accuracy(logits, label_ids)
			

	# Report the final accuracy for this validation run.
	avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
	print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

	# Calculate the average loss over all of the batches.
	avg_val_loss = total_eval_loss / len(test_dataloader)

	# Measure how long the validation run took.
	validation_time = format_time(time.time() - t0)

	print("  Validation Loss: {0:.2f}".format(avg_val_loss))
	print("  Validation took: {:}".format(validation_time))
	return preds

def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", required=False, default="./Dataloaders",
											help="Location of data files.")
	parser.add_argument("--trained-model-dir", required=False, default="./Output",
											help="Location of the saved trained model.")
	parser.add_argument("--output-dir", required=False, default="./Output",
											help="Output directory to save the predictions.")

	args = parser.parse_args()
	data_dir = args.data_dir
	output_dir = args.output_dir

	if os.path.isdir(args.trained_model_dir):
		trained_model_dir = args.trained_model_dir
	else:
		raise Exception('Trained model directory "{args.trained_model_dir}" does not exist.')

	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)


	test_dataloader = torch.load(os.path.join(data_dir, 'test.pth'))
	labels = test_dataloader.dataset.tensors[2].tolist()
	trained_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(trained_model_dir, "trained_model"))

	preds = test(trained_model, test_dataloader)
	preds = np.concatenate(preds).argmax(axis=1)
	np.save(os.path.join(output_dir, 'predictions.npy'), preds) # save
	scores = calculate_score(preds, labels)
	print(scores)

if __name__ == "__main__":
  main()
