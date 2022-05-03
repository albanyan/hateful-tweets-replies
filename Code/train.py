import argparse
import torch
import os
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import random
import numpy as np

EPOCHS = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
import time
import datetime

TASK_MODE ='all' # Can take values ["all", "main", "reply"]

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(train_dataloader, validation_dataloader, model, scheduler, optimizer):
	# This training code is based on the `run_glue.py` script here:
	# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

	# Set the seed value all over the place to make this reproducible.
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	# We'll store a number of quantities such as training and validation loss, 
	# validation accuracy, and timings.
	training_stats = []

	# Measure how long the training epoch takes.
	total_t0 = time.time()
	best_score = 1
	best_score1 = 0

	# For each epoch...
	for epoch_i in range(0, EPOCHS):

			# Perform one full pass over the training set.
			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
			print('Training...')

			# Measure how long the training epoch takes.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_train_loss = 0

			model.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):

					# Progress update every 40 batches.
					if step % 40 == 0 and not step == 0:
							# Calculate elapsed time in minutes.
							elapsed = format_time(time.time() - t0)
							
							# Report progress.
							print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

					# Unpack this training batch from our dataloader. 
					#
					# As we unpack the batch, we'll also copy each tensor to the GPU using the 
					# `to` method.
					#
					# `batch` contains three pytorch tensors:
					#   [0]: input ids 
					#   [1]: attention masks
					#   [2]: labels 
					b_input_ids = batch[0].to(DEVICE)
					b_input_mask = batch[1].to(DEVICE)
					b_labels = batch[2].to(DEVICE)

					# Always clear any previously calculated gradients before performing a
					# backward pass. PyTorch doesn't do this automatically because 
					# accumulating the gradients is "convenient while training RNNs". 
					# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
					model.zero_grad()        

					# Perform a forward pass (evaluate the model on this training batch).
					# The documentation for this `model` function is here: 
					# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
					# It returns different numbers of parameters depending on what arguments
					# arge given and what flags are set. For our useage here, it returns
					# the loss (because we provided labels) and the "logits"--the model
					# outputs prior to activation.
					outputs = model(b_input_ids, 
															token_type_ids=None, 
															attention_mask=b_input_mask, 
															labels=b_labels)
					loss = outputs.loss
					logits = outputs.logits
					# Accumulate the training loss over all of the batches so that we can
					# calculate the average loss at the end. `loss` is a Tensor containing a
					# single value; the `.item()` function just returns the Python value 
					# from the tensor.
					total_train_loss += loss.item()

					# Perform a backward pass to calculate the gradients.
					loss.backward()

					# Clip the norm of the gradients to 1.0.
					# This is to help prevent the "exploding gradients" problem.
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

					# Update parameters and take a step using the computed gradient.
					# The optimizer dictates the "update rule"--how the parameters are
					# modified based on their gradients, the learning rate, etc.
					optimizer.step()

					# Update the learning rate.
					scheduler.step()

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)            
			
			# Measure how long this epoch took.
			training_time = format_time(time.time() - t0)

			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epcoh took: {:}".format(training_time))
					
			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.

			print("")
			print("Running Validation...")

			t0 = time.time()

			# Put the model in evaluation mode--the dropout layers behave differently
			# during evaluation.
			model.eval()

			# Tracking variables 
			total_eval_accuracy = 0
			total_eval_loss = 0
			nb_eval_steps = 0

			# Evaluate data for one epoch
			for batch in validation_dataloader:
					
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

					# Calculate the accuracy for this batch of test sentences, and
					# accumulate it over all batches.
					total_eval_accuracy += flat_accuracy(logits, label_ids)
					

			# Report the final accuracy for this validation run.
			avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
			print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

			# Calculate the average loss over all of the batches.
			avg_val_loss = total_eval_loss / len(validation_dataloader)
			
			# Measure how long the validation run took.
			validation_time = format_time(time.time() - t0)
			
			print("  Validation Loss: {0:.2f}".format(avg_val_loss))
			print("  Validation took: {:}".format(validation_time))

			# Record all statistics from this epoch.
			training_stats.append(
					{
							'epoch': epoch_i + 1,
							'Training Loss': avg_train_loss,
							'Valid. Loss': avg_val_loss,
							'Valid. Accur.': avg_val_accuracy,
							'Training Time': training_time,
							'Validation Time': validation_time
					}
			)

			if avg_val_loss < best_score:
					print("save the model...")
					# torch.save(model.roberta.state_dict(),'/content/gdrive/MyDrive/Dataset/BERTweet models/test1000.pb')
					# torch.save(model.roberta.state_dict(),'/content/gdrive/MyDrive/Dataset/BERTweet models/BERTtweet_'+dataset+'_'+sub_dataset+'_'+Q+'.pb')
					best_score = avg_val_loss

			# if avg_val_accuracy > best_score1:
			#     print("save the model...")
			#     torch.save(model.roberta.state_dict(),'/content/gdrive/MyDrive/Dataset/BERTweet models/test1001.pb')
			#     # torch.save(model.roberta.state_dict(),'/content/gdrive/MyDrive/Dataset/BERTweet models/BERTtweet_'+dataset+'_'+sub_dataset+'_'+Q+'.pb')
			#     best_score1 = avg_val_accuracy


	print("")
	print("Training complete!")

	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-name", required=True,
											help='The model name, either bert or bertweet.')
	parser.add_argument("--data-dir", required=False, default="./Dataloaders",
											help="Location of data files.")
	parser.add_argument("--output-dir", required=False, default="./Output",
											help="Output directory to save the trained model.")

	args = parser.parse_args()
	data_dir = args.data_dir
	output_dir = args.output_dir

	if args.model_name.lower() not in ["bert", "bertweet"]:
		raise Exception(f"Model name must be either bert or bertweet.")
	else:
		model_name = args.model_name.lower()

	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
		
	train_dataloader = torch.load(os.path.join(data_dir, 'train.pth'))
	validation_dataloader = torch.load(os.path.join(data_dir, 'valid.pth'))
	
	if model_name == "bert":
		# Load BertForSequenceClassification, the pretrained BERT model with a single 
		# linear classification layer on top. 
		model = AutoModelForSequenceClassification.from_pretrained(
				"bert-base-uncased", 
				num_labels = 2, # The number of output labels--2 for binary classification.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		model.classifier.out_features = 128
		model.classifier.add_module(module=torch.nn.ReLU(), name='Activation')

		model.classifier.add_module(module=torch.nn.Linear(128, 2), name="additional")
		model.classifier.additional.add_module(module=torch.nn.Softmax(), name='Activation')
	else:
		# Load BertForSequenceClassification, the pretrained BERTweet model
		model = AutoModelForSequenceClassification.from_pretrained(
				"vinai/bertweet-base", 
				num_labels = 2, # The number of output labels--2 for binary classification.
				output_attentions = False, # Whether the model returns attentions weights.
				output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		model.classifier.dense.out_features = 128
		model.classifier.dense.add_module(module=torch.nn.ReLU(), name='Activation')

		model.classifier.out_proj.in_features = 128
		model.classifier.out_proj.add_module(module=torch.nn.Softmax(), name='Activation')

	model.to(DEVICE)

	optimizer = AdamW(model.parameters(),
						lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
						eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
						)

	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * EPOCHS

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, # Default value in run_glue.py
												num_training_steps = total_steps
												)

	train(train_dataloader, validation_dataloader, model, scheduler, optimizer)
	model.save_pretrained(os.path.join(output_dir, "trained_model"))
	
if __name__ == "__main__":
  main()
