import argparse
import paddle
import paddle.autograd
import os
import os.path as osp
import numpy as np
import paddle.optimizer as optim
from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator
from paddle.fluid import ParamAttr
import paddle.nn as nn 
import paddle.fluid as fluid
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss, CrossEntropyLoss
import paddle.nn.functional as F

#from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet



IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 0

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19
RESTORE_FROM = './model/pretrained.pdparams'
#RESTORE_FROM = './snapshots/GTA2Cityscapes_CVPR_Syn0820_Wg00005weight005_dampingx2/GTA5_36000.pth' #For retrain
#RESTORE_FROM_D = './snapshots/GTA2Cityscapes_CVPR_Syn0820_Wg00005weight005_dampingx2/GTA5_36000_D.pth' #For retrain

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)
#print(PREHEAT_STEPS)
POWER = 0.9
RANDOM_SEED = 1234

SOURCE = 'SYNTHIA'
TARGET = 'cityscapes'
SET = 'train'

if SOURCE == 'GTA5':
    INPUT_SIZE_SOURCE = '1280,720'
    DATA_DIRECTORY = './data/GTA5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 40
    Epsilon = 0.4
elif SOURCE == 'SYNTHIA':
    INPUT_SIZE_SOURCE = '1024,512'
    DATA_DIRECTORY = './data/SYNTHIA/RAND_CITYSCAPES'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 10
    Epsilon = 0.4
    
INPUT_SIZE_TARGET = '1024,512'
DATA_DIRECTORY_TARGET = './data/Cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
#device = paddle.set_device('gpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #print(label.shape)
    label=paddle.to_tensor(label, dtype='float32',stop_gradient=False)
    #label =paddle.to_tensor(label, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
    #criterion = CrossEntropy2d(NUM_CLASSES)
    criterion = CrossEntropyLoss(NUM_CLASSES)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.set_lr(lr)
    #if len(optimizer.param_groups) > 1:
    optimizer.set_lr(lr * 10)


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.set_lr(lr)
    #if len(optimizer.param_groups) > 1:
    optimizer.set_lr(lr * 10)


def weightmap(pred1, pred2):
    a_1 = paddle.sum((pred1 * pred2), 1)
    a = paddle.reshape(a_1, [1, 1, pred1.shape[2], pred1.shape[3]])
    b = paddle.reshape(paddle.norm(pred1, 2, 1) * paddle.norm(pred2, 2, 1),[1, 1, pred1.shape[2], pred1.shape[3]])
    output = 1.0 -  a/b 
    
    return output


def main():
    """Create the model and start the training."""
    
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    #cuDNN.enabled = False
    
    # Create Network
    model = Res_Deeplab(num_classes=args.num_classes)

    print(args.restore_from)
    saved_state_dict = paddle.load(args.restore_from)
    new_params = model.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'layer5':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
     
    model.set_state_dict(new_params)
    
    model.train()
  
    model_D = FCDiscriminator(num_classes=args.num_classes)
# =============================================================================
#    #for retrain     
#    saved_state_dict_D = paddle.load(RESTORE_FROM_D)
#    model_D.set_state_dict(saved_state_dict_D)
# =============================================================================
    
    model_D.train()
    #model_D.cuda(args.gpu)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    if args.source == 'GTA5':
        trainloader = paddle.io.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,)
    else:
        #print(args.data_list)
        trainloader = paddle.io.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trainloader_iter = enumerate(trainloader)

    targetloader = paddle.io.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=True, mirror=True, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    targetloader_iter = enumerate(targetloader)

    optimizer = optim.Momentum(learning_rate=args.learning_rate, momentum=args.momentum, parameters = model.parameters(), weight_decay=args.weight_decay)
    optimizer.clear_grad()

    optimizer_D = optim.Adam(learning_rate=args.learning_rate_D, beta1=0.9, beta2=0.99, epsilon=1e-08, parameters=model_D.parameters())
    optimizer_D.clear_grad()

    bce_loss = paddle.nn.BCEWithLogitsLoss() 
    weighted_bce_loss = WeightedBCEWithLogitsLoss()
    #criterion = nn.functional.cross_entropy()

    interp_source = paddle.nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True,data_format='NCHW')
    interp_target = paddle.nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True,data_format='NCHW')
    
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):
        optimizer.clear_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.clear_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
        damping = (1 - i_iter/NUM_STEPS)
        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            #param.requires_grad = False
            param.stop_gradient = True

        # Train with Source
        #for _, batch in trainloader_iter：
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s=paddle.to_tensor(images_s, dtype='float32',stop_gradient=False)
        
        pred_source1, pred_source2 = model(images_s)
        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)
        
        #Segmentation Loss

        loss_seg = paddle.to_tensor((loss_calc(pred_source1, labels_s) + loss_calc(pred_source2, labels_s)),stop_gradient=False)
        #loss_seg = paddle.to_tensor((nn.functional.cross_entropy(pred_source1, labels_s) + nn.functional.cross_entropy(pred_source2, labels_s)),stop_gradient=False)
        loss_seg.backward()
        #print(loss_seg.stop_gradient)
        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, _, _, _ = batch
      
        images_t=paddle.to_tensor(images_t, dtype='float32',stop_gradient=False)
        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, axis = 1), F.softmax(pred_target2, axis = 1))
        
        D_out = interp_target(model_D(F.softmax(pred_target1 + pred_target2, axis = 1)))
        
        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            v1 = paddle.full_like(D_out,source_label)
            v1 = paddle.to_tensor(v1, stop_gradient=False)
            loss_adv = weighted_bce_loss(D_out, v1, weight_map, Epsilon, Lambda_local)
        else:
            v2 = paddle.full_like(D_out,source_label)
            v2 = paddle.to_tensor(v2, stop_gradient=False)
            loss_adv = bce_loss(D_out, v2)
                   
        loss_adv = paddle.to_tensor(loss_adv * Lambda_adv * damping,stop_gradient=False)
        loss_adv.backward()
        #print(loss_adv.stop_gradient)
        #Weight Discrepancy Loss 
        W5 = None
        W6 = None
        if args.model == 'ResNet':

            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = paddle.reshape(w5,[-1])
                    W6 = paddle.reshape(w6,[-1])
                else:
                    W5 = paddle.concat((W5, paddle.reshape(w5,[-1])), 0)
                    W6 = paddle.concat((W6, paddle.reshape(w6,[-1])), 0)
        
        loss_weight = (paddle.matmul(W5, W6) / (paddle.norm(W5) * paddle.norm(W6)) + 1) # +1 is for a positive loss
        loss_weight = paddle.to_tensor(loss_weight * Lambda_weight * damping * 2,stop_gradient=False)
        loss_weight.backward()
        #print(loss_weight.stop_gradient)
        #======================================================================================
        # train D
        #======================================================================================
        
        # Bring back Grads in D
        for param in model_D.parameters():
            #param.requires_grad = True 
            param.stop_gradient = False 
            
        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()
        
        D_out_s = interp_source(model_D(F.softmax(pred_source1 + pred_source2, axis = 1)))

        #v3 = paddle.full_like(D_out_s,source_label)
        v3 = paddle.to_tensor(paddle.full_like(D_out_s,source_label), stop_gradient=False)
        loss_D_s = paddle.to_tensor(bce_loss(D_out_s,v3) ,stop_gradient=False)

        loss_D_s.backward()
        #print(loss_D_s.stop_gradient)
        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()
        
        D_out_t = interp_target(model_D(F.softmax(pred_target1 + pred_target2, axis = 1)))
        
        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            v4 = paddle.full_like(D_out_t,target_label)
            v4 = paddle.to_tensor(v4, stop_gradient=False)

            loss_D_t = paddle.to_tensor(weighted_bce_loss(D_out_t, v4, weight_map, Epsilon, Lambda_local),stop_gradient=False)
        else:
            v5 = paddle.full_like(D_out_t,target_label)
            v5 = paddle.to_tensor(v5, stop_gradient=False)
            loss_D_t = paddle.to_tensor(bce_loss(D_out_t,v5),stop_gradient=False)
            
        loss_D_t.backward()
        #print(loss_D_t.stop_gradient)
        optimizer.step()
        optimizer_D.step()
        

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(i_iter, args.num_steps, loss_seg.item(), loss_adv.item(), loss_weight.item(), loss_D_s.item(), loss_D_t.item()))
        f_loss = open(osp.join(args.snapshot_dir,'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(loss_seg.item(), loss_adv.item(), loss_weight.item(), loss_D_s.item(), loss_D_t.item()))
        f_loss.close()
        
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            paddle.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            paddle.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            paddle.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pdparams'))
            paddle.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pdparams'))

if __name__ == '__main__':
    main()
