import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
       
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, 256)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
  

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding_size = 256
        self.hidden_feature_size = 512
        self.vocabulary_size = vocab_size
        self.number_layers = num_layers

        # embedding layer that turns words into a vector of a specified size.
        self.word_embeddings = nn.Embedding(self.vocabulary_size, self.embedding_size)

        # the RNN model takes embedded word vectors as inputs and outputs hidden states of size hidden_dim
        self.rnn_model = nn.GRU(self.embedding_size, self.hidden_feature_size, num_layers=self.number_layers, batch_first=True)

        # the linear layer that maps the hidden state output to the vocabulary size.
        self.hidden2vocabulary = nn.Linear(self.hidden_feature_size, self.vocabulary_size)
    
    def forward(self, features, captions):
        # Define the feed-forward behavior of the model. Create embedded word vectors for each word in a sentence
        captions = captions[:, :-1]     # discarding end token
        caption_embeddings = self.word_embeddings(captions)

        model_input = torch.cat((features.unsqueeze(dim=1), caption_embeddings), dim=1)
        model_output, _ = self.rnn_model(model_input)
        vocabulary_probabilities = self.hidden2vocabulary(model_output)
        return vocabulary_probabilities

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        generated_caption_indices = list()
        for i in range(max_len):
            rnn_output, states = self.rnn_model(inputs, states)
            vocabulary_probabilities = self.hidden2vocabulary(rnn_output)

            _, probable_word_index = vocabulary_probabilities.max(dim=2)
            generated_caption_indices.append(probable_word_index.item())

            inputs = self.word_embeddings(probable_word_index)

        return generated_caption_indices