#The code here is for prediction. 
#The code here allows prediction to be deployed as an api.

#To import libraries
from flask import Flask, render_template, request, jsonify
import torch
import torchvision
import numpy as np
import cv2
import face_recognition
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch import nn
from torchvision import models
from google_drive_downloader import GoogleDriveDownloader as gdd

#The architecture of the model
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

#To perform prediction
sm = nn.Softmax()
def prediction(model,img, path='./'):
  fmap,logits = model(img.to('cuda'))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  return [int(prediction.item()),confidence]

#To validate the data
class validation(Dataset):
    def __init__(self,video_names,sequenceLength = 60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequenceLength
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        videoPath = self.video_names[idx]
        frames = []
        for i,frame in enumerate(self.frameExtract(videoPath)):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
              break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frameExtract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return "Hello"
    

@app.route('/video/<string:n>')
def video(n):
    try:
        gdd.download_file_from_google_drive(file_id=n,dest_path='./video.mp4',overwrite=True)
        videoPath='./video.mp4'
        cap=cv2.VideoCapture(videoPath)
        if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))>18000):
            jsonify({"Video": "is too lengthy", "Confidence":0})
        videoPath=[videoPath]
        model = Model(2).cuda()
        pathToModel = './checkpoint.pt'
        model.load_state_dict(torch.load(pathToModel))
        model.eval()
        size = 112
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]  
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        videoDataset = validation(videoPath,sequenceLength = 10,transform = transform)
        pred=prediction(model,videoDataset[0],'./')
        p=''
        if pred[0]==1:
            p="REAL"
            print(p)
        else:
            p="FAKE"
            print(p)
        print(pred[1])
        return jsonify({"Video":p, "Confidence":pred[1]})
    except:
        return jsonify({"Video": "Not a video file", "Confidence":0})

if __name__== '__main__':
    app.run(port=3000, debug=True)