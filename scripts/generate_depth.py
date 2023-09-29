import torch
import json
import cv2
import os
import random
import numpy as np

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
device = torch.device('cpu') 
midas.to(device)
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
save_item = {}
def generate_depth(path, output_name, img_dir):
        with open(path, 'r') as f:
                data = json.load(f)

        print("data: ", data[0]['img_name'], " len: ", len(data))
        i = 0
        for item in data: 
        
               
                interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, 
                                        cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                
                img_path = os.path.join(img_dir, item["img_name"])
                img_name = item["img_name"]

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (192,192),
                                interpolation=random.choice(interp_methods))
                with torch.no_grad(): 
                        imgbatch = transform(img).to('cpu')
                        prediction = midas(imgbatch)
                        prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size = img.shape[:2], 
                        mode='bicubic', 
                        align_corners=False
                        ).squeeze()
                        depth = prediction.cpu().numpy()
                        depth = [depth]
                if(np.shape(depth) != (1,192,192)):
                        print("Depth shape: ", np.shape(depth), " image name: ", img_name)
                #depth = np.array(depth, dtype=np.float32)
                save_item[img_name] = depth
                
                i += 1
                percentage = (i/len(data)) * 100
                print(" Calculating depth of dataSet: ",(int(percentage)), "/100%    |    Total: ",i," / ", len(data), end='\r')
                #print("Total: ",i," / ", len(data), end='\r')
                #depths.append(save_item)
        ######
        with open(output_name,'w') as f: 
                print("writing output json file") 
                json.dump(save_item, f, ensure_ascii=False, cls=NpEncoder)
        print('Total write ', len(save_item))


img_dir = "/mnt/fast0/asmi20/movenet.depth/data/cropped/imgs"

train_path = "/mnt/fast0/asmi20/movenet.depth/data/cropped/train2017.json"
train_output = "/mnt/fast0/asmi20/movenet.depth/data/cropped/train_depth.json"

generate_depth(train_path,train_output,img_dir)

val_path = "/mnt/fast0/asmi20/movenet.depth/data/cropped/val2017.json"
val_output = "/mnt/fast0/asmi20/movenet.depth/data/cropped/val_depth.json"

generate_depth(val_path,val_output,img_dir)