import torch
import argparse
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import LatentDataset
from net.classifier import classifier

parser = argparse.ArgumentParser()
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_path', type=str, default='./data/celebahq_anno.npy', help='label file path')
args = parser.parse_args()

attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    isTrain = True
    epoch = 100
    dataset = LatentDataset(args.latent_path, args.label_path,True)
    data_iter = DataLoader(dataset,batch_size,shuffle=isTrain,num_workers = 8)
    print(len(data_iter))
    net = classifier(fmaps=[9216, 2048, 512, 40],activ='leakyrelu').to(device)
    net.train()
    parameters = list(net.parameters())
    optimizer = torch.optim.Adam(parameters,lr=1e-4,weight_decay = 0.0005)
    BCEloss = nn.BCEWithLogitsLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch_n in range(epoch):
        print('epoch',epoch_n)
        for i,(origin_img,label) in enumerate(tqdm(data_iter)):
            origin_img = origin_img.to(device)
            label = label.to(device)

            out_label = net(origin_img.reshape(origin_img.shape[0],-1))
            out_label = torch.sigmoid(out_label)
            loss = BCEloss(out_label,label.float())
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 ==0:
                print('temp loss:',loss)
        scheduler.step()
    log_dir = os.path.join('./models')
    torch.save(net.state_dict(),'{:s}/new_latent_classifier_epoch_{:d}.pth'.format(log_dir, epoch ))            
    test(net.state_dict(),device)

def test(net_dict,device):
    dataset = LatentDataset(args.latent_path, args.label_path,False)
    data_iter = DataLoader(dataset,batch_size = 1,shuffle=False)
    net = classifier(fmaps=[9216, 2048, 512, 40],activ='leakyrelu').to(device)
    net.load_state_dict(net_dict,strict=False)
    net.eval()
    BCEloss = nn.BCEWithLogitsLoss(reduction='none')
    total_num = 0
    true_num = 0
    real_lbl=[]
    pred_lbl=[]
    valid_num=[]
    for (origin_img,label) in tqdm(data_iter):
        with torch.no_grad():
            origin_img = origin_img.to(device)
            label = label.to(device)
            out_label = net(origin_img.reshape(origin_img.shape[0],-1))
            out_label = torch.sigmoid(out_label).round().long()
            #print(out_label,label)
            #if total_num >10:
            #    return 
            if out_label.equal(label):
                true_num+=1
            total_num+=1
            acc = true_num/total_num
            #print('total acc:{:.4f}'.format(acc))

            real_lbl.append(label)
            pred_lbl.append(out_label.data)
            valid_num.append((label == out_label).long())
    print(label.shape)
    real_lbl = torch.cat(real_lbl, dim=0)
    pred_lbl = torch.cat(pred_lbl, dim=0)
    valid_num = torch.cat(valid_num, dim=0)
    print(real_lbl.shape,pred_lbl.shape,valid_num.shape)
    T_num = torch.sum(real_lbl, dim=0)
    F_num = total_num - T_num
    print(T_num.shape,F_num.shape)
    pred_T_num = torch.sum(pred_lbl, dim=0)
    pred_F_num = total_num - pred_T_num
    print(pred_T_num.shape,pred_F_num.shape)
    True_Positive = torch.sum(valid_num * real_lbl, dim=0)
    True_Negative = torch.sum(valid_num * (1 - real_lbl), dim=0)

    # Recall
    recall_T = True_Positive.float()/(T_num.float() + 1e-8)
    recall_F = True_Negative.float()/(F_num.float() + 1e-8)

    # Precision
    precision_T = True_Positive.float()/(pred_T_num.float() + 1e-8)
    precesion_F = True_Negative.float()/(pred_F_num.float() + 1e-8)

    # Accuracy
    valid_num = torch.sum(valid_num, dim=0)
    accuracy = valid_num.float()/total_num

    for i in range(40):
        print('%0.3f'%recall_T[i].item(), '%0.3f'%recall_F[i].item(), '%0.3f'%precision_T[i].item(), '%0.3f'%precesion_F[i].item(), '%0.3f'%accuracy[i].item())


if __name__=='__main__':
    main()
    #test(torch.load('./models/new_latent_classifier_epoch_10.pth'),'cuda')