
import os,argparse
import random
import numpy as np
from lib import init, Data, MoveNet, Task

from config import cfg




def main(cfg):


    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    
    

    data = Data(cfg)
    #data_loader = data.getEvalDataloader()
    train_loader, val_loader = data.getTrainValDataloader()


    run_task = Task(cfg, model)


    run_task.modelLoad("output/e145_valacc0.79951.pth")
    run_task.evaluate(val_loader)



    # model = MoveNet(num_classes=cfg["num_classes"],
    #                 width_mult=cfg["width_mult"],
    #                 mode='test')
    
    

    # data = Data(cfg)
    # data_loader = data.getEvalDataloader()


    # run_task = Task(cfg, model)


    # run_task.modelLoad("output/e92_valacc0.98326.pth")
    # run_task.evaluateTest(data_loader)


if __name__ == '__main__':
    main(cfg)