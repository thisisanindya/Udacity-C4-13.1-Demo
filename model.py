import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
'''
Example of sample featues - Kept for reference.


type(features): <class 'torch.Tensor'>
features.shape: torch.Size([10, 256])
features ---> 
 tensor([[-0.1343,  0.3846,  0.1602,  ..., -0.0185, -0.0641, -0.7531],
        [ 0.3519, -0.0889, -0.0615,  ..., -0.1911,  0.5807, -0.0074],
        [ 0.5066, -0.5379,  0.2319,  ..., -0.5771, -0.3000, -0.6492],
        ...,
        [ 0.6051,  0.1673,  0.0215,  ...,  0.1066, -0.1792, -0.3509],
        [-0.5125,  0.8928, -0.1867,  ...,  0.4215,  0.5452, -0.3878],
        [-0.1629, -0.7519, -0.2278,  ...,  0.2252, -0.0283, -0.7034]],
       grad_fn=<AddmmBackward0>)

This is a PyTorch tensor with shape [batch_size, embed_size]
''' 

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        #super(DecoderRNN, self).__init__()
        super().__init__()
        print("Running updated code...")
        # TODO: Complete this function
        self.embed_size = embed_size        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers        
        
        ## LSTM Layer
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first = True)
        
        ## Fully connected linear layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        ## Embedding Layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        

    def forward(self, features, captions, hidden_state=None):
        ## embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        
        ## Embed te captions
        embedded_captions = self.embed(captions)
        
        ## Isolate elements
        image_features = features.unsqueeze(1)
        
        ## Concat features and caption embeddings
        concat_embeddings = torch.cat((image_features, embedded_captions[:, :-1,:]), dim=1)
        
        ## Execute LSTM cell
        output, hidden_state = self.lstm(concat_embeddings, hidden_state)
        
        ## Apply fullly connected layer
        output = self.fc(output)
        
        ## Return the output
        return output
                                  
    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence
