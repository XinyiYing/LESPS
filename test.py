import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch LESPS test")
parser.add_argument("--model_names", default=['ACM', 'ALCNet', 'DNANet'], type=list, help="model_name: 'ACM', 'ALCNet', 'DNANet'")
parser.add_argument("--pth_dirs", default=['SIRST3/ACM_full.pth.tar','SIRST3/ACM_LESPS_centroid.pth.tar','SIRST3/ACM_LESPS_coarse.pth.tar',
                                           'SIRST3/ALCNet_full.pth.tar','SIRST3/ALCNet_LESPS_centroid.pth.tar','SIRST3/ALCNet_LESPS_coarse.pth.tar',
                                           'SIRST3/DNANet_full.pth.tar','SIRST3/DNANet_LESPS_centroid.pth.tar','SIRST3/DNANet_LESPS_coarse.pth.tar',], 
                                            type=list, help="checkpoint dir, default=None")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K',], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./dataset/SIRST3', type=str, help="train_dataset_dir")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
            
            ### save img
            model_name = opt.pth_dir.split('/')[-1] .split('.')[0]
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,:,:size[0],:size[1]]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + model_name ):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + model_name )
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + model_name + '/' + img_dir[0] + '.png')  
        
        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()
        print("pixAcc, mIoU:\t" + str(results1))
        print("PD, FA:\t" + str(results2))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open('./test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    for pth_dir in opt.pth_dirs:
        opt.train_dataset_name = pth_dir.split('/')[0]
        print(pth_dir)
        opt.f.write(pth_dir + '\n')
        for dataset_name in opt.dataset_names:
            opt.test_dataset_name = dataset_name
            opt.pth_dir = opt.save_log + pth_dir
            print(opt.test_dataset_name)
            opt.f.write(opt.test_dataset_name + '\n')
            for model_name in opt.model_names:
                if model_name in pth_dir:
                    opt.model_name = model_name
            test()
        print('\n')
        opt.f.write('\n')
    opt.f.close()
