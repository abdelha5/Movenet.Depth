import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch.nn.functional as F

from lib.task.task_tools import getSchedu, getOptimizer,movenetDecode,clipGradient
from lib.loss.movenet_loss import MovenetLoss
from lib.utils.utils import printDash
from lib.utils.metrics import myAcc
from lib.utils.metrics import myAcc, getmAP
import sys
sys.path.append('../env/lib/python3.8/site-packages')
import wandb





class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg

        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        ############################################################
        # loss
        self.loss_func = MovenetLoss()
        
        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'], 
                                    self.model, 
                                    self.cfg['learning_rate'], 
                                    self.cfg['weight_decay'])

        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)
        

    def train(self, train_loader, val_loader):

        for epoch in range(self.cfg['epochs']):

            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader, epoch)

        self.onTrainEnd()


    def predict(self, data_loader, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.eval()
        correct = 0

        with torch.no_grad():

            for (img, img_name) in data_loader:

                # if "yoga_img_483" not in img_name[0]:
                #     continue

                
                # print(img.shape, img_name)
                img = img.to(self.device)

                output = self.model(img)
                #print(len(output))
                


                pre = movenetDecode(output, None, mode='output')


                basename = os.path.basename(img_name[0])
                img = np.transpose(img[0].cpu().numpy(),axes=[1,2,0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h,w = img.shape[:2]

                for i in range(len(pre[0])//2):
                    x = int(pre[0][i*2]*w)
                    y = int(pre[0][i*2+1]*h)
                    cv2.circle(img, (x, y), 3, (255,0,0), 2)

                cv2.imwrite(os.path.join(save_dir,basename), img)
                

                ## debug
                heatmaps = output[0].cpu().numpy()[0]
                centers = output[1].cpu().numpy()[0]
                regs = output[2].cpu().numpy()[0]
                offsets = output[3].cpu().numpy()[0]

                #print(heatmaps.shape)
                hm = cv2.resize(np.sum(heatmaps,axis=0),(192,192))*255
                cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_heatmaps.jpg"),hm)
                img[:,:,0]+=hm
                cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_img.jpg"), img)
                cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_center.jpg"),cv2.resize(centers[0]*255,(192,192)))
                cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_regs0.jpg"),cv2.resize(regs[0]*255,(192,192)))
                


    def label(self, data_loader, save_dir):
        self.model.eval()
        

        txt_dir = os.path.join(save_dir, 'txt')
        show_dir = os.path.join(save_dir, 'show')

        with torch.no_grad():

            for (img, img_path) in data_loader:
                #print(img.shape, img_path)
                img_path = img_path[0]
                basename = os.path.basename(img_path)

                img = img.to(self.device)

                output = self.model(img)
                #print(len(output))
                


                pre = movenetDecode(output, None, mode='output')[0]
                #print(pre)
                with open(os.path.join(txt_dir,basename[:-3]+'txt'),'w') as f:
                    f.write("7\n")
                    for i in range(len(pre)//2):
                        vis = 2
                        if pre[i*2]==-1:
                            vis=0
                        line = "%f %f %d\n" % (pre[i*2], pre[i*2+1], vis)
                        f.write(line)

                

                img = np.transpose(img[0].cpu().numpy(),axes=[1,2,0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h,w = img.shape[:2]

                for i in range(len(pre)//2):
                    x = int(pre[i*2]*w)
                    y = int(pre[i*2+1]*h)
                    cv2.circle(img, (x, y), 3, (255,0,0), 2)

                cv2.imwrite(os.path.join(show_dir,basename), img)
                

                #b
                

    def exam(self, data_loader, save_dir):
        self.model.eval()


        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):
                
                if batch_idx%5000 == 0:
                    print('Finish ',batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                
                pre = movenetDecode(output, kps_mask,mode='output')
                gt = movenetDecode(labels, kps_mask,mode='label')
                
                #n
                acc = myAcc(pre, gt)
                

                # if 'mypc1_full_1180' in img_names[0]:
                if 0/7<sum(acc)/len(acc)<=5/7:
                # if sum(acc)/len(acc)==1: 
                    # print(pre)
                    # print(gt)
                    # print(acc)
                    img_name = img_names[0]
                    # print(img_name)

                    basename = os.path.basename(img_name)
                    save_name = os.path.join(save_dir, basename)


                    hm = cv2.resize(np.sum(output[0][0].cpu().numpy(),axis=0),(192,192))*255
                    cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_hm_pre.jpg"),hm)

                    hm = cv2.resize(np.sum(labels[0,:7,:,:].cpu().numpy(),axis=0),(192,192))*255
                    cv2.imwrite(os.path.join(save_dir,basename[:-4]+"_hm_gt.jpg"),hm)



                    img = np.transpose(imgs[0].cpu().numpy(),axes=[1,2,0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    h,w = img.shape[:2]


                    for i in range(len(gt[0])//2):
                        x = int(gt[0][i*2]*w)
                        y = int(gt[0][i*2+1]*h)
                        cv2.circle(img, (x, y), 5, (0,255,0), 3)


                        x = int(pre[0][i*2]*w)
                        y = int(pre[0][i*2+1]*h)
                        cv2.circle(img, (x, y), 3, (0,0,255), 2)

                    cv2.imwrite(save_name, img)

                
                    #bb
                
    def evaluate1(self, data_loader):
        self.model.eval()

        correct = 0
        total = 0
        prediction = []
        ground = []
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):
                if batch_idx%100 == 0:
                    print('Finish ',batch_idx)
                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)
                t_prid = []
                t_gt = []
                
                # for i in range(len(imgs)):
                #     print("this: ", np.shape(imgs[i]))
                #     output = self.model(imgs[i])
                #     prid = movenetDecode(output, kps_mask,mode='output')
                #     gt = movenetDecode(labels[i], kps_mask,mode='label')
                #     pred : {
                #         "img_name" : imgs[i],
                #         "keypoints" : prid
                #     }
                #     ground_truth : {
                #         "img_name" : imgs[i],
                #         "keypoints" : gt
                #     }
                #     t_prid.append(pred)
                #     t_gt.append(ground_truth)
                output = self.model(imgs)
                pre = movenetDecode(output, kps_mask,mode='output')
                gt = movenetDecode(labels, kps_mask,mode='label')
                acc = myAcc(pre, gt)
                
                correct += sum(acc)
                total += len(acc)
    
        acc = correct/total
        print('[Info] acc: {:.3f}% \n'.format(100. * acc))
            
                # coco_eval = COCOeval(t_gt, t_prid, 'keypoints')
                # coco_eval.evaluate()
                # coco_eval.accumulate()
                # coco_eval.summarize()
               
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

    def evaluate(self, data_loader):
        self.model.eval()

        correct = 0
        total = 0
        prediction = []
        ground = []
        joints_acc = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        total_acc = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):
                
                if batch_idx%100 == 0:
                    print('Finish ',batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)
                # coco_eval = COCOeval(labels, output, 'keypoints')
                # coco_eval.evaluate()
                # coco_eval.accumulate()
                # coco_eval.summarize()
                
                pre = movenetDecode(output, kps_mask,mode='output')
                gt = movenetDecode(labels, kps_mask,mode='label')
              
                # pr = pre[1].tolist()
                # gr = gt[1].tolist()   
                # prediction = prediction + pr
                # ground = ground + gr
                #n
                acc = myAcc(pre, gt)
                for i in range(len(acc)):
                    joints_acc[i] = joints_acc[i] + acc[i] 
                    total_acc[i] = total_acc[i] + 1
                #correct += sum(acc)
                #total += len(acc)
        elapsed_time = time.time() - start_time
        print("Time: ", elapsed_time, " for ", len(data_loader), " frames")
        print(joints_acc)
        print("total acc: ", total_acc)
        for i in range(len(joints_acc)):
            joints_acc[i] = joints_acc[i]/total_acc[i]
        print('[Info] mAP of each keypoint: ', joints_acc)
        # acc = correct/total
        # print('[Info] acc: {:.3f}% \n'.format(100. * acc))


    def evaluateTest(self, data_loader):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):
                
                if batch_idx%100 == 0:
                    print('Finish ',batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs).cpu().numpy()
                # print(output)
                # b

                pre = []
                for i in range(7):
                    if output[i*3+2]>0.1:
                        pre.extend([output[i*3],output[i*3+1]])
                    else:
                        pre.extend([-1,-1])
                pre = np.array([pre])
                
                # pre = movenetDecode(output, kps_mask,mode='output')
                gt = movenetDecode(labels, kps_mask,mode='label')
                # print(pre, gt)
                # b
                #n
                acc = myAcc(pre, gt)
                
                correct += sum(acc)
                total += len(acc)

        acc = correct/total
        print('[Info] acc: {:.3f}% \n'.format(100. * acc))


################ 
    def onTrainStep(self,train_loader, epoch):
        
        self.model.train()
        correct = 0
        count = 0



        right_count = np.array([0]*self.cfg['num_classes'], dtype=np.int64)
        total_count = 0

        # midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        # midas.to('cpu')
        # midas.eval()
        # transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        # transform = transforms.small_transform 

        for batch_idx, (imgs, labels, kps_mask,img_names) in enumerate(train_loader):

            # if '000000242610_0' not in img_names[0]:
            #     continue
            #print("img: ", np.shape(imgs[0]), len(imgs[0]))

            labels = labels.to(self.device)
            imgs = imgs.to(self.device)
            kps_mask = kps_mask.to(self.device)
            #print("img: ", imgs)
            output = self.model(imgs)


            heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss = self.loss_func(output, labels, kps_mask)

            total_loss = heatmap_loss+center_loss+regs_loss+offset_loss+bone_loss 

            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])

            
            self.optimizer.zero_grad()#把梯度置零
            total_loss.backward() #计算梯度
            self.optimizer.step() #更新参数


            ### evaluate

            pre = movenetDecode(output,kps_mask, mode='output')
            
            gt = movenetDecode(labels,kps_mask, mode='label')


            # hm = cv2.resize(np.sum(labels[0,:7,:,:].cpu().detach().numpy(),axis=0),(192,192))*255
            # cv2.imwrite(os.path.join("output/show_img",os.path.basename(img_names[0])[:-4]+"_gt.jpg"),hm)
            #bb
            # print(pre.shape, gt.shape)
            # b
            acc = myAcc(pre, gt)
            #map = getmAP(pre, gt)

            right_count += acc
            total_count += labels.shape[0]


            if batch_idx%self.cfg['log_interval']==0:
                print('\r', 
                        '%d/%d '
                        '[%d/%d] '
                        'loss: %.4f '
                        '(hm_loss: %.3f '
                        'b_loss: %.3f '
                        'c_loss: %.3f '
                        'r_loss: %.3f '
                        'o_loss: %.3f) - '
                        'acc: %.4f       ' % (epoch+1,self.cfg['epochs'],
                                        batch_idx, len(train_loader.dataset)/self.cfg['batch_size'],
                                        total_loss.item(),
                                        heatmap_loss.item(),
                                        bone_loss.item(),
                                        center_loss.item(),
                                        regs_loss.item(),
                                        offset_loss.item(),
                                        np.mean(right_count/total_count)),
                                        end='',flush=True)
            # break
        print()


    def onTrainEnd(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()


    def onValidation(self, val_loader, epoch):

        num_test_batches = 0.0
        self.model.eval()
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Implementation of movenet with Depth 2",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.001,
            "architecture": "CNN",
            "dataset": "COCO_2017",
            "epochs": 150,
            }
        )


        right_count = np.array([0]*self.cfg['num_classes'], dtype=np.int64)
        total_count = 0
        predictions = []
        ground_truths = []
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(val_loader):
                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss = self.loss_func(output, labels, kps_mask)
                total_loss = heatmap_loss+center_loss+regs_loss+offset_loss+bone_loss

                ### evaluate
                # acc1 = myAcc(heatmap2locate(output[0].detach().cpu().numpy()), 
                #                 heatmap2locate(labels[:,:7,:,:].detach().cpu().numpy()))
                pre = movenetDecode(output, kps_mask,mode='output')
                gt = movenetDecode(labels, kps_mask,mode='label')
                acc = myAcc(pre, gt)
                prediction = prediction + pre
                ground_truths = ground_truths + gt
                # right_count1 += acc1
                right_count += acc
                total_count += labels.shape[0]

                # break
            
            wandb.log({"Epoch": epoch ,"Acc": np.mean(right_count/total_count), "Total Loss": total_loss, "Heatmap Loss": heatmap_loss, "Center Loss": center_loss, "Bone Loss": bone_loss, "Regs Loss": regs_loss,"Offset Loss": offset_loss})
            
            print('LR: %f - '
                ' [Val] loss: %.5f '
                '[hm_loss: %.4f '
                'bone_loss: %.4f '
                'center_loss: %.4f '
                'r_loss: %.4f '
                'offset_loss: %.4f] - '
                'mAP: %.4f          ' % ( 
                                self.optimizer.param_groups[0]["lr"],
                                total_loss.item(),
                                heatmap_loss.item(),
                                bone_loss.item(),
                                center_loss.item(),
                                regs_loss.item(),
                                offset_loss.item(),
                                np.mean(right_count/total_count)),
                                )
            print()

        if 'default' in self.cfg['scheduler']:
            self.scheduler.step(np.mean(right_count/total_count))
        else:
            self.scheduler.step()

        save_name = 'e%d_valacc%.5f.pth' % (epoch+1,np.mean(right_count/total_count))
        self.modelSave(save_name)

        


    def onTest(self):
        self.model.eval()
        
        #predict
        res_list = []
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list





    def modelLoad(self,model_path, data_parallel = False):
        self.model.load_state_dict(torch.load(model_path), strict=True)
        
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)


    def modelSave(self, save_name):
        torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], save_name))
        #print("Save model to: ",save_name)
