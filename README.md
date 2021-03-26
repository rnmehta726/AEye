# AEye - Visual Assistance App
An intuitive app for the visually impaired to assist them in navigating their envoironment through the implementation of an image-captioning model in PyTorch.

## How It Works

### Image Captioning Model
An image-captioning model was trained in PyTorch on Microsoft's COCO dataset to understand images many different environments and produce accurate textual descriptions of the image. The model consists of two parts, the pre-trained ResNet50 convolutional neural network and a recurrent neural network that is one layer deep. The ResNet50 model is used to extract features from the image, and the RNN takes those features and generates a textual description of the image. The model can be found in [model.py](https://github.com/rnmehta726/AEye/blob/main/image_captioning_model/model.py)

### Mobile App
The mobile app was created to offer an intuitive, affordable, and accessible tool for the visually impaired to use. The app has one screen with a button that fills the entire screen. On the press of this button, an image is captured through the camera and this image is sent to the api found in [web_api](https://github.com/rnmehta726/AEye/tree/main/web_api) using http.post to be run through the image-captioning model. The api then returns the generated sentence of the image back to the app where it is converted to audio for the user to hear. The app is not published, but it does work for iOS and Android because Flutter was used.

![image](https://user-images.githubusercontent.com/64166777/112535128-e8217c80-8d79-11eb-9786-a214e49292c9.png)

### Web App
The web app functions as an api for the mobile app and was built using Flask. The web app takes the image, preprocesses it to be run through the image-captioning model, and runs it through the image captioning model to get a textual description of the image. The description is then sent back to the app to to read aloud to the user. 
