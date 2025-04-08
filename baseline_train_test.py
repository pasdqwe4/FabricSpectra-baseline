import time
import torch.nn
import torch.nn.functional as F
import torch.utils.data as Data
from data_utils.data_process import *
import model_utils
from sklearn import metrics
import os
import shutil
from model.baselines import *
from model.swin_v2 import *
from model.tsai import *
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from chronos import ChronosPipeline

model_utils.setup_seed(0)
# model = OmniScaleCNN(c_in = 2, c_out=1024 ,seq_len=200 ).cuda()

'''Adopt model to experiment'''
# model = ChronosPipeline.from_pretrained(
#         "amazon/chronos-t5-tiny",
#         device_map="cuda",
#         torch_dtype=torch.bfloat16,
#     )
#model = gmlp()
#model = convnextv2_tiny()
#model = tss()
#model = Swin_t()
#model = gmlp()
model = Swinv2_t()

'''For multi-gpu'''
# model = nn.DataParallel(model)

optimizer = optim.Lamb(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.1)

def main():
    ##name of save path (saving model)
    save_path_acc = "save_path"
    check_dir(save_path_acc)

    ### Just foi initializing
    best_hamming = 1000
    best_acc_val = 0
    best_mse = 1000

    best_a0 = 0
    best_a1 = 0
    best_a2 = 0
    best_a3 = 0
    best_a5 = 0
    best_a10 = 0

    root = "root_path"
    # train path
    trainpath = root + '/' + 'train_x.npy'
    train_label_path = root + '/' + 'train_y.npy'
    train_comp_label_path = root + '/' + 'train_c.npy'
    #test path
    testpath = root + '/' + 'test_x.npy'
    test_label_path = root + '/' + 'test_y.npy'
    test_comp_label_path = root + '/' + 'test_c.npy'

    train_set = NIRDataset_train(trainpath, train_label_path, train_comp_label_path)
    val_set = NIRDataset_test(testpath, test_label_path, test_comp_label_path)

    batch_size = 1024
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=False)

    epochs = 100
    for epoch in range(epochs):
        #train
        train(train_loader, epoch)
        scheduler.step()

        acc_val, mse, hamming , a0, a1, a2, a3, a5, a10= validate(val_loader, epoch)

        #evaluation metrics
        if acc_val >= best_acc_val:
            best_acc_val = acc_val
            ## save best model
            save_checkpoint({
                'state_dict': model.state_dict(),
            },  path=save_path_acc)

        if mse <= best_mse:
            best_mse = mse
            # best_epoch_mse
            save_checkpoint_mse({
                'state_dict': model.state_dict(),
            }, path=save_path_acc)

        if hamming <= best_hamming:
            best_hamming = hamming
        if a0 > best_a0:
            best_a0 = a0
        if a1 > best_a1:
            best_a1 = a1
        if a2 > best_a2:
            best_a2 = a2
        if a3 > best_a3:
            best_a3 = a3
        if a5 > best_a5:
            best_a5 = a5
        if a10 > best_a10:
            best_a10 = a10

        save_checkpoint_time({

                'state_dict': model.state_dict(),

            }, path=save_path_acc)

        print("##########################################################################################################################")
        print('Multi-label result')
        print('******** Best acc: {}, current acc: {}'.format(best_acc_val, acc_val))
        print('******** Best hamming loss: {}, current hamming loss: {}'.format(best_hamming, hamming))
        print('Regression result')
        print('******** best 0% acc: {}, current 0% acc: {}'.format(best_a0, a0))
        print('******** best 1% acc: {}, current 1% acc: {}'.format(best_a1, a1))
        print('******** best 2% acc: {}, current 2% acc: {}'.format(best_a2, a2))
        print('******** best 3% acc: {}, current 3% acc: {}'.format(best_a3, a3))
        print('******** best 5% acc: {}, current 5% acc: {}'.format(best_a5, a5))
        print('******** best 10% acc: {}, current 10% acc: {}'.format(best_a10, a10))
        print('******** Best mse: {}, current mse: {}'.format(best_mse,mse))
        print("##########################################################################################################################")

    print('Train Finished!')
    print('Best val_acc:', best_acc_val)

def train(train_loader, cur_ep):

    model.train()
    t = time.time()

    for batch_idx, batch in enumerate(train_loader):
        x, y, comp= batch
        x = x.cuda()
        y = y.cuda()
        comp = comp.cuda()

        optimizer.zero_grad()

        ## predicted labels for each classes
        mlc, regress = model(x)

        mlc_loss = F.multilabel_soft_margin_loss(mlc, y)
        comp_loss = F.l1_loss(regress, comp)

        total_loss = mlc_loss + comp_loss

        final_pred_reg = regress.cpu().detach().numpy()
        final_pred = threshold_predictions(mlc, 0.5)
        final_pred = final_pred.cpu().detach().numpy()

        acc_train = metrics.accuracy_score(y.cpu().detach().numpy(), final_pred, normalize=True,
                                           sample_weight=None)

        mse = metrics.mean_squared_error(comp.cpu().detach().numpy(), final_pred_reg)

        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print("####################################################################################")
            print('Epoch: {:04d}'.format(cur_ep + 1),
                  'iteration:{:04d}'.format(batch_idx + 1)),
            print('loss_train_mlc: {:.8f}'.format(mlc_loss.item()),
                  'loss_train_regress: {:.8f}'.format(comp_loss.item()))
            print("------------------------------------------------")
            print('total_loss: {:.8f}'.format(total_loss.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'mse_train: {:.4f}'.format(mse),
                  'time: {:.4f}s'.format(time.time() - t))
            print("####################################################################################")
        t = time.time()

def validate(val_loader, cur_ep):
    model.eval()
    t = time.time()

    total_acc = 0
    total_hamming = 0
    total_mse = 0

    avg_acc = 0
    avg_mse = 0
    avg_hamming = 0

    count = 0

    total_pred_reg = np.zeros(([0, 12]))
    total_truth_reg = np.zeros(([0, 12]))

    for batch_idx, batch in enumerate(val_loader):
        x, y, comp= batch

        x = x.cuda()
        y = y.cuda()
        comp = comp.cuda()

        mlc, regress = model(x)

        final_pred_reg = regress.cpu().detach().numpy()
        final_pred = threshold_predictions(mlc, 0.5)
        final_pred = final_pred.cpu().detach().numpy()

        mse = metrics.mean_squared_error(comp.cpu().detach().numpy(), final_pred_reg)

        acc_val = metrics.accuracy_score(y.cpu().detach().numpy(), final_pred, normalize=True, sample_weight=None)
        hamming = metrics.hamming_loss(y.cpu().detach().numpy(), final_pred)

        n = x.shape[0]
        total_acc += acc_val * n
        total_mse += mse * n
        total_hamming += hamming * n

        count += n

        avg_acc = total_acc / count
        avg_hamming = total_hamming / count
        avg_mse = total_mse / count

        total_pred_reg = np.append(total_pred_reg, final_pred_reg, axis=0)
        total_truth_reg = np.append(total_truth_reg, comp.cpu().detach().numpy(), axis=0)

        if batch_idx % 10 == 0:
            print('Epoch: {:04d}'.format(cur_ep + 1),
              'iteration:{:04d}'.format(batch_idx + 1)),
            print('mse_val: {:.4f}'.format(mse),
              'acc_val: {:.4f}'.format(acc_val),
                'hamming_val: {:.4f}'.format(hamming),
              'time: {:.4f}s'.format(time.time() - t))
        t = time.time()

    avg_acc_0 = cal_error_acc(total_pred_reg, total_truth_reg, 0)
    avg_acc_1 = cal_error_acc(total_pred_reg, total_truth_reg, 1)
    avg_acc_2 = cal_error_acc(total_pred_reg, total_truth_reg, 2)
    avg_acc_3 = cal_error_acc(total_pred_reg, total_truth_reg, 3)
    avg_acc_5 = cal_error_acc(total_pred_reg, total_truth_reg, 5)
    avg_acc_10 = cal_error_acc(total_pred_reg, total_truth_reg, 10)

    return avg_acc, avg_mse, avg_hamming, avg_acc_0, avg_acc_1, avg_acc_2, avg_acc_3, avg_acc_5, avg_acc_10
def check_dir(dir_path):
    """���Ŀ¼�Ƿ���ڣ��������򴴽�"""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def save_checkpoint(state, path, filename='checkpoint.pth.tar'):
    """ģ�ͱ���"""
    save_filename = path + '/' + filename
    torch.save(state, save_filename)

    shutil.copyfile(save_filename, path + '/model_best.pth.tar')

def save_checkpoint_mse(state, path, filename='checkpoint_mae.pth.tar'):
    """ģ�ͱ���"""
    save_filename = path + '/' + filename
    torch.save(state, save_filename)

    shutil.copyfile(save_filename, path + '/model_best_mse.pth.tar')

def save_checkpoint_time(state, path, filename='checkpoint_time.pth.tar'):
    """ģ�ͱ���"""
    save_filename = path + '/' + filename
    torch.save(state, save_filename)

def cal_error_acc(output, label, threshold):
    n = output.shape[0]
    b = 0
    for i in range (n):
        if (np.absolute(output[i]-label[i]) <= threshold).all():
            b +=1

    if b ==0:
        return 0
    return b/n

def threshold_predictions(logits, threshold=0.5):
    """Applies a threshold to the logits to obtain binary predictions.

    Args:
        logits: A tensor of shape (batch_size, num_classes).
        threshold: The threshold value.

    Returns:
        A tensor of shape (batch_size, num_classes) containing binary predictions.
    """
    predictions = torch.sigmoid(logits) > threshold
    return predictions.int()

if __name__ == '__main__':

    main()
    pass