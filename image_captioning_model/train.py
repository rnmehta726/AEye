import torch
import torch.nn as nn
import sys
from torchvision import transforms
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time


batch_size = 32          
vocab_threshold = 6     
vocab_from_file = True            
num_epochs = 3            
save_every = 1             
print_every = 100         
log_file = 'training_log.txt'   


transform_train = transforms.Compose([ 
    transforms.Resize(256),                          
    transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = EncoderCNN()
decoder = DecoderRNN(vocab_size, num_layers=1)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

params = list(decoder.parameters()) + list(encoder.embed.parameters())

optimizer = torch.optim.Adadelta(params)
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size) 

f = open(log_file, 'w')

for epoch in range(1, num_epochs+1):
    
    for i_step in range(1, total_step+1):
        
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
        
        # Get training statistics.
        perplexity_val = np.exp(loss.item())
        stats = f'Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], Loss: {loss.item()}, Perplexity: {perplexity_val}')
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)         
        if perplexity_val < 6.5:
            torch.save(decoder, os.path.join('models', f'decoder-{epoch}-{i_step}-{perplexity_val}.pt'))
            torch.save(encoder, os.path.join('models', f'encoder-{epoch}-{i_step}-{perplexity_val}.pt'))
            
    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder, os.path.join('models', f'decoder-{epoch}.pt'))
        torch.save(encoder, os.path.join('models', f'encoder-{epoch}.pt'))
    #         learning_rate_scheduler.step()

# Close the training log file.
f.close()