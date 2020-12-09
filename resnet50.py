import os
import cv2 #import OpenCV
import torch
import torch.utils.data as data
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import DataLoader



def random_crop(img,mask, crop_size=(10, 10)):
    img = img.copy()
    mask = mask.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    mask = mask[y:y+crop_size[0], x:x+crop_size[1]]
    return img,mask
def flip_img(img,mask):
    #flipcode = 0: flip vertically
    #flipcode > 0: flip horizontally
    #flipcode < 0: flip vertically and horizontally
    flipCode = np.random.choice([0,1,-1],1)[0]   
    return cv2.flip(img, flipCode),cv2.flip(mask, flipCode)
def rotate_img(img,mask):
    rotateCode = np.random.choice([cv2.ROTATE_90_CLOCKWISE,
                                   cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_180],1)[0]
    return cv2.rotate(img, rotateCode),cv2.rotate(mask, rotateCode)
def gaussian_noise(img,mask, mean=0, sigma=15):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 255.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 0
    noise[mask_overflow_lower] = 0
    img += noise
    return img,mask
def distort(img,mask):
    img_dist = img.copy()
    mask_dist = mask.copy()
    orientation = np.random.choice(['ver', 'hor'],1)[0]
    func = np.random.choice([np.sin,np.cos],1)[0]
    x_scale = np.random.choice([0.01, 0.02, 0.03, 0.04],1)[0]
    y_scale = np.random.choice([2, 4, 6, 8, 10],1)[0]
    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))
    
    for c in range(3):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i] = np.roll(img[:, i], shift(i))
                mask_dist[:, i] = np.roll(mask[:, i], shift(i))
            else:
                img_dist[i, :] = np.roll(img[i, :], shift(i))
                mask_dist[i, :] = np.roll(mask[i, :], shift(i))
            
    return img_dist,mask_dist

aug_pipeline=[flip_img,rotate_img,gaussian_noise,distort]

def img_augmentation(img, mask):
    np.random.choice(aug_pipeline, np.random.randint(5, size=1), replace=False)
    pipeline = np.random.randint(2, size=5)
    for process in aug_pipeline:
        img,mask = process(img,mask)
    return img, mask

#------------------------------------------------------------------------------------------------
class TrainDataset(data.Dataset):
    def __init__(self, root=''):
        super(TrainDataset, self).__init__()
        self.img_files = glob(os.path.join(root,'image','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root,'mask',basename[:-4]+'_mask.png'))
            

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')     
            label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32')
            data,label = img_augmentation(data,label)
            # process data
            rgb = cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)
            #rgb = np.transpose(rgb, [2,0,1])
            resized = cv2.resize(rgb, (225,225), interpolation = cv2.INTER_NEAREST)
            normalized = (resized - np.min(resized))/(np.max(resized)-np.min(resized))
            normalized[:,:,-3] = (normalized[:,:,-1]-0.485)/0.229
            normalized[:,:,-2] = (normalized[:,:,-1]-0.456)/0.224
            normalized[:,:,-1] = (normalized[:,:,-1]-0.406)/0.225

            # process label
            label_array=np.zeros((96,96,4))
            label_array[:,:,0]=(label==0).astype(int)
            label_array[:,:,1]=(label==1).astype(int)
            label_array[:,:,2]=(label==2).astype(int)
            label_array[:,:,3]=(label==3).astype(int)

            resized_mask = cv2.resize(label_array, (225,225), interpolation = cv2.INTER_NEAREST)
            
            return torch.from_numpy(normalized).float(), torch.from_numpy(resized_mask).long()

    def __len__(self):
        return len(self.img_files)
    
class ValidationDataset(data.Dataset):
    def __init__(self, root=''):
        super(ValidationDataset, self).__init__()
        self.img_files = glob(os.path.join(root,'image','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root,'mask',basename[:-4]+'_mask.png'))
            

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')         
            label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32')
            # process data
            rgb = cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)
            #rgb = np.transpose(rgb, [2,0,1])
            resized = cv2.resize(rgb, (225,225), interpolation = cv2.INTER_NEAREST)
            normalized = (resized - np.min(resized))/(np.max(resized)-np.min(resized))
            normalized[:,:,-3] = (normalized[:,:,-1]-0.485)/0.229
            normalized[:,:,-2] = (normalized[:,:,-1]-0.456)/0.224
            normalized[:,:,-1] = (normalized[:,:,-1]-0.406)/0.225

            # process label
            label_array=np.zeros((96,96,4))
            label_array[:,:,0]=(label==0).astype(int)
            label_array[:,:,1]=(label==1).astype(int)
            label_array[:,:,2]=(label==2).astype(int)
            label_array[:,:,3]=(label==3).astype(int)

            resized_mask = cv2.resize(label_array, (225,225), interpolation = cv2.INTER_NEAREST)
            
            return torch.from_numpy(normalized).float(), torch.from_numpy(resized_mask).long()

    def __len__(self):
        return len(self.img_files)

class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root,'image','*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')
            # process data
            rgb = cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)
            #rgb = np.transpose(rgb, [2,0,1])
            resized = cv2.resize(rgb, (225,225), interpolation = cv2.INTER_NEAREST)
            normalized = (resized - np.min(resized))/(np.max(resized)-np.min(resized))
            normalized[:,:,-3] = (normalized[:,:,-1]-0.485)/0.229
            normalized[:,:,-2] = (normalized[:,:,-1]-0.456)/0.224
            normalized[:,:,-1] = (normalized[:,:,-1]-0.406)/0.225
            
            return torch.from_numpy(normalized).float()

    def __len__(self):
        return len(self.img_files)
#------------------------------------------------------------------------------------------------
def dice_coef(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    loss_sum = torch.tensor(0.0,requires_grad=True)
    for i in range(y_true.shape[0]):
        for j in range(4):
            loss_sum = torch.add(loss_sum,dice_coef(y_true[i][j],y_pred[i][j]))
    #print(loss_sum)
    # return mean loss of batch
    return 4-(loss_sum/y_true.shape[0])
#------------------------------------------------------------------------------------------------------




# Pyramid Pooling Module
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        # bins : size of the ppm output
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.1, classes=4, zoom_factor=8, use_ppm=True, criterion=dice_coef_loss):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        resnet = torchvision.models.resnet50(pretrained=True)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
      
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)


        # since feature maps of resnet18 = 512
        fea_dim = 2048
        
        # Pyramid Pooling Module
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1),
            nn.Softmax2d()
        )
        if self.training:
            self.aux = nn.Sequential(
                # fit the size to layer3 of resnet
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1),
                nn.Softmax2d()
            )

    def forward(self, x, y=None):
        x_size = x.size()
        #print(x.shape)
        assert (x_size[2]-1) % 8 == 0
        assert (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #print('3:',x.shape)
        #print('x_temp:',self.layer3(x).shape)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        #print('x4:',x.shape)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            #print('aux:',aux.shape,aux.dtype)
            #print(aux.dtype)
            
            return (x, aux)
        else:
            return x

#------------------------------------------------------------------------------------------------
def save_checkpoint(state, filename='model_best.pth.tar'):
    torch.save(state, filename)
#------------------------------------------------------------------------------------------------

def main():
    
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # check whether a GPU is available

    if torch.cuda.is_available():
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())

    data_path = './data/train'
    val_path = './data/val'

    num_workers = 4
    #torch.set_num_threads(4)

    lr = 1e-3
    momentum= 0.9
    weight_decay= 0.0001

    epochs = 80

    load_weight=False


    batch_size = 16
    # define auxilury loss weight
    aux_weight = 0.4

    # define data
    train_set = TrainDataset(data_path)
    validation_set = ValidationDataset(val_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, 
                                      batch_size=batch_size, shuffle=True)
                                     #multiprocessing_context=torch.multiprocessing.get_context('spawn'))
    validation_data_loader = DataLoader(dataset=validation_set, num_workers=num_workers, 
                                       batch_size=batch_size, shuffle=False)
                                      #multiprocessing_context=torch.multiprocessing.get_context('spawn'))

    #define model
    model = PSPNet().to(device)
    if load_weight:
        model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])

    # define loss function
    loss_fn = dice_coef_loss

    # define optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr,
    #                            momentum=momentum, weight_decay=weight_decay)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    # define logger
    losses = list()
    sum_val_losses = 0.0
    val_losses = list()
    lowest_val_loss = 1000.0

    for epoch in range(epochs):
        print("epoch:",epoch+1)
            
        # Fetch images and labels.
        for iteration, sample in enumerate(training_data_loader):


            print("iteration:",iteration)

            # training mode
            model.train()

            img, mask = sample
            img = img.to(device)
            mask = mask.to(device)

            #show_image_mask(img[0,...].squeeze(), mask[0,...].squeeze()) #visualise all data in training set
            #plt.pause(1)
            # reshape data from NHWC to NCHW
            img = img.permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2)
            #print('mask:',mask.shape,mask.dtype)
            #print(mask.dtype)

            # Write your FORWARD below
            # Note: Input image to your model and ouput the predicted mask and Your predicted mask should have 4 channels
            output, aux = model(img, mask)
            main_loss = loss_fn(output,mask)
            
            aux_loss = loss_fn(aux,mask)
            loss = torch.add(main_loss, torch.mul(aux_weight, aux_loss))

            losses.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation on the end of epoch

            with torch.no_grad():       # fix the model's params
                for val_iteration, sample in enumerate(validation_data_loader):
                    #print('Validation:',iteration)
                    model.eval()    # set model to evaluation mode
                    img, mask = sample
                    img = img.to(device)
                    mask = mask.to(device)
                    # reshape data from NHWC to NCHW
                    img = img.permute(0, 3, 1, 2)
                    mask = mask.permute(0, 3, 1, 2)
                    output = model(img, mask)
                    main_loss = loss_fn(output,mask)
                    
                    #output = output.max(1)[1]
                    #output = output.float()
                    #print('output',output.shape, mask.dtype)
                    #print(output[0][0])
                    #print('mask',mask.shape, mask.dtype)
                    #val_loss = loss_fn(output,sample)
                    sum_val_losses+=(main_loss.detach().cpu().numpy())
                val_losses.append(sum_val_losses/len(validation_data_loader))
                sum_val_losses=0

        #print("end epoch:",epoch+1)
        #print()


    # Then write your BACKWARD & OPTIMIZE below
    # Note: Compute Loss and Optimize
     # remember best prec@1 and save checkpoint
        is_best = lowest_val_loss > val_losses[-1]
        #is_best = True
        lowest_val_loss = min(lowest_val_loss, val_losses[-1])
        if is_best:        
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_val_loss': lowest_val_loss,
            })
            
        out_losses = np.array(losses)
        out_val_losses = np.array(val_losses)

        with open('loss.npy', 'wb') as f:
            np.save(f, out_losses)
            np.save(f, out_val_losses)


if __name__ == "__main__":
    main()
