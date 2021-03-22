from flask import Flask, redirect, url_for, request
import base64
from io import BytesIO
import os
import pickle
import json
from torchvision import transforms
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)


@app.route("/", methods = ['GET', 'POST'])
def home():
	if request.method == "POST":
		data = request.get_json(force=True)
		im = Image.open(BytesIO(base64.b64decode(data['img'])))
		im.save('accept.jpg', 'JPEG')

		device = torch.device("cpu")
		transform_test = transforms.Compose([
			transforms.Resize(256),  # smaller edge of image resized to 256
			transforms.RandomCrop(224),  # get 224x224 crop from random location
			transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
    		transforms.ToTensor(),  # convert the PIL Image to a tensor
    		transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                        		(0.229, 0.224, 0.225))])

		encoder = torch.load('encoder1.pt')
		encoder.eval()
		decoder = torch.load('decoder1.pt')
		decoder.eval()

		encoder.to(device)
		decoder.to(device)

		PIL_image = Image.open('accept.jpg').convert('RGB')
		final_image = transform_test(PIL_image)
		final_image = final_image.unsqueeze(0)
		print(final_image.shape)
		final_image = final_image.to(device)

		caption = get_prediction(image=final_image, encoder=encoder, decoder=decoder)
		print(caption)

		os.remove('accept.jpg')
		return f"Success! <h1>Yay</h1>"
	else:
   		return "Post has not yet been called <h1>GET</h1>"

def clean_sentence(output):
	idx2word = {}
	with open('vocab.pkl', 'rb') as f:	
			vocab = pickle.load(f)
			idx2word = vocab.idx2word
	sentence = ''
	for each_caption_index in output:
		each_caption_word = idx2word[each_caption_index]
		if each_caption_word == '<start>':
			continue
		if each_caption_word == '<end>':
			break

		sentence += ' ' + each_caption_word
	return sentence

def get_prediction(image, encoder, decoder):
	features = encoder(image).unsqueeze(1)
	output = decoder.sample(features)
	sentence = clean_sentence(output)
	return sentence    

if __name__ == "__main__":
	app.run()