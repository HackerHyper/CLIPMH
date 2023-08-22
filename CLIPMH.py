import torch.nn as nn
import torch
import torchvision


import torch
import clip
from PIL import Image

import torch.nn.functional as F




device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hash_bit = args.hash_bit

        
        
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        
        

        # fusion

        self.fc3 = nn.Linear(512, 256)
        
        self.activation3 = nn.Tanh()
       
        self.image_out = nn.Sequential(
            self.fc3,self.activation3)

        self.text_convert = nn.Linear(512, 256)
        self.text_act = nn.Tanh()
        
        self.text_out = nn.Sequential(self.text_convert, self.text_act)

        

        self.coff = nn.Sequential(
            nn.Linear(256*2,256*2, bias=False),
            nn.Sigmoid())

       
       
        self.final_hash = nn.Linear(256*2, self.hash_bit)
        self.fus_act3 = nn.Tanh()
        self.fus_dropout = nn.Dropout(0.1)

       

        self.hash_output = nn.Sequential(
            self.final_hash, self.fus_act3)

        self.iter_num = 0
        self.scale = 1


    def forward(self, image,text):

        # image = preprocess(Image.open("data/flickr25k/images/im1.jpg")).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

        
        text = clip.tokenize(text).to(device)

        
        image_features = self.model.encode_image(image).float()
        text_features = self.model.encode_text(text).float()

        # print(image_features.shape)

        # print(text_features.shape)

        image_out = self.image_out(image_features)

      
        text_out = self.text_out(text_features)

        
        feat_concat = torch.cat((image_out, text_out),1)

        coff = self.coff(feat_concat)

        feat_fusion = coff*feat_concat 
        

        return self.hash_output(feat_fusion)


        

