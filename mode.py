import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
#from skimage.color import rgb2ycbcr
#from skimage.measure import compare_psnr
import time

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(args, rank, world_size):       #python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 main.py

    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = args.in_memory,transform = transform)
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True,sampler=sampler,  num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num, scale=args.scale)
    
    
    if args.fine_tuning:        
        generator.load_state_dict(torch.load(args.generator_path))
        print("pre-trained model is loaded")
        print("path : %s"%(args.generator_path))
        
    generator = generator.to(device)
    generator = DDP(generator, device_ids=[rank])
    generator.train()
    
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr = 1e-4)
        
    pre_epoch = 2
    fine_epoch = 0
    
    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1
        

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())

            print('=========')
        data =[]
        fig = go.FigureWidget(data=[{'type': 'scatter'}])
        #fig.add_scatter()
        fig
        data.append(loss.item())
        for i in range(len(data)):
            time.sleep(0.3)
            fig.data[0].y = data[:i+1]
        if pre_epoch % 1400 ==0:
            torch.save(generator.state_dict(), './model/pre_trained_model_%03d.pt'%pre_epoch)

        
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()
    
    discriminator = Discriminator(patch_size = args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()
    
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    
    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)
    
    while fine_epoch < args.fine_train_epoch:
        
        scheduler.step()
        
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)
                        
            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = args.feat_layer)
            
            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat)**2)
            
            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss
            
            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            #y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            #y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            #psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0) #######################
            #psnr_list.append(psnr)
            
        fine_epoch += 1

        if fine_epoch % 2 == 0:
            print(fine_epoch)
            print(g_loss.item())
            print(d_loss.item())
            print('=========')

        if fine_epoch % 200 ==0:
            #torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            #torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)
            torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)
    cleanup()

# In[ ]:

def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path), map_location=torch.device('cpu'), strict=False)
    generator = generator.to(device)
    generator.eval()
    
    f = open('./result.txt', 'w')
    psnr_list = []
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w *args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            
            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
            psnr_list.append(psnr)
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)
        return psnr
        #f.write('avg psnr : %04f' % np.mean(psnr_list))


def test_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = testOnly_data(LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master address
    os.environ['MASTER_PORT'] = '12345'      # Set the master port
    os.environ['RANK'] = '0'                 # Set the rank of the current process
    os.environ['WORLD_SIZE'] = '1'           # Set the total number of processes
    dist.init_process_group(backend='gloo')
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path, map_location=torch.device('cpu')))
    generator = nn.parallel.DistributedDataParallel(generator)
    generator = generator.to(device)
    generator.eval()
    image_timings = []
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            image_start_time = time.time()
            lr = te_data['LR'].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)
            image_end_time = time.time()
            image_elapsed_time = (image_end_time - image_start_time) * 1000
            image_timings.append(image_elapsed_time)
            print("Image {}: Elapsed time: {:.2f} milliseconds".format(i + 1, image_elapsed_time))
    average_image_timing = np.mean(image_timings)
    print("Average timing for each image: {:.2f} milliseconds".format(average_image_timing))


import onnx
import onnxruntime as ort

## **Testing ONNX model**
def test_only_onnx(args):
    # Load the ONNX model
    onnx_model = onnx.load(args.generator_path)
    
    # Initialize ONNX Runtime session
    ort_session = ort.InferenceSession(args.generator_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = testOnly_data(LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    image_timings = []
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            image_start_time = time.time()
            lr = te_data['LR'].to(device)

            # Run inference using ONNX Runtime
            ort_inputs = {ort_session.get_inputs()[0].name: lr.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            output = torch.tensor(ort_outs[0])

            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1, 2, 0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png' % i)
            image_end_time = time.time()
            image_elapsed_time = (image_end_time - image_start_time) * 1000
            image_timings.append(image_elapsed_time)
            print("Image {}: Elapsed time: {:.2f} milliseconds".format(i + 1, image_elapsed_time))

    average_image_timing = np.mean(image_timings)
    print("Average timing for each image: {:.2f} milliseconds".format(average_image_timing))


