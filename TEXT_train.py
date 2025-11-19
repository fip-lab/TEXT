import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, best_result, best_result_val
from tensorboardX import SummaryWriter
from model.TEXT import TEXT
from core.metric import MetricsTop
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



val_results_list = []
test_results_list = []

val_global_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}}


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()


def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
    return res


def main(opt, device):

    # print("======================= 打印参数信息 =======================")
    # for key, value in vars(opt).items():
    #     print(key, value)

    print("\n=================== 打印其他信息 ===========================")
    print("device: {}".format(device))

    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))

    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root, opt.project_name, opt.datasetName)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    print("datapath: {}".format(opt.dataPath))
    if  os.path.exists(opt.dataPath):
        print("Data path exists!")

    # ===================================================
    time.sleep(0.1)
    print("===================================================", opt.datasetName, opt.fusion_layer_depth, opt.num_experts, opt.top_k, opt.capacity_factor, opt.dropout)
    model = TEXT(dataset=opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth, num_experts=opt.num_experts, top_k=opt.top_k, capacity_factor=opt.capacity_factor, dropout=opt.dropout).to(device)
    print("=================================================== parameter num: ", count_parameters(model))
    # print("=================================================== model: ", model)


    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)

    print("\n")
    epoch_pbar = tqdm(range(1, opt.n_epochs+1))


    per_train_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}}

    for epoch in epoch_pbar:
        train_metric = train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        eval_metric = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, save_path, metrics)
        if opt.is_test is not None:
            test_metric = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metrics, save_path)
        scheduler_warmup.step()

        epoch_pbar.set_description(opt.datasetName) 
        epoch_pbar.set_postfix({'test_metric': '{}'.format(test_metric)})



        acc2_val = 'Has0_acc_2' if 'Has0_acc_2' in eval_metric else 'Mult_acc_2' 
        
        # Update the best results for per_train_acc2_best
        if epoch == 1 or eval_metric[acc2_val] >= per_train_acc2_best['val_result'][acc2_val]:
            per_train_acc2_best['epoch'] = epoch
            per_train_acc2_best['val_result'] = eval_metric
            per_train_acc2_best['test_result'] = test_metric



        # Update the best results for val_global_acc2_best
        if eval_metric[acc2_val] >= val_global_acc2_best['val_result'][acc2_val]:
            val_global_acc2_best['epoch'] = epoch
            val_global_acc2_best['val_result'] = eval_metric
            val_global_acc2_best['test_result'] = test_metric

            # Check if the save path exists, if it does, delete the model
            if os.path.exists(save_path):
                for file in os.listdir(save_path):
                    if file.endswith('.pth'):
                        os.remove(os.path.join(save_path, file))
            save_model(save_path, epoch, model, optimizer)

        if epoch == opt.epochs:
            break
        

    # 写入结果到文件
    with open(opt.result_txtPath, 'a') as f:
        f.write("======:   ")
        for key, value in per_train_acc2_best['test_result'].items():
            f.write(f"{value:<15.4f}")
        f.write(f"{opt.datasetName:<8}【{opt.note}】{per_train_acc2_best['epoch']}\n")


    writer.close()






def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):    
    train_pbar = enumerate(train_loader)

    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    #  text, audio, visual, summary_clue, audio_clue, visual_clue
    for cur_iter, data in train_pbar:

        text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)

        loss = loss_fn(output, label)
        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    writer.add_scalar('train/loss', losses.value_avg, epoch)
    return train_results





def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics):
    eval_pbar = enumerate(eval_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in eval_pbar:
            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)  # 前向传播

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())


        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = metrics(pred, true)
        val_results_list.append(eval_results)

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)

    return eval_results





def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics, save_path):
    test_pbar = enumerate(test_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            # img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            # output = model(img, audio, text)
            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        test_results_list.append(test_results)

        writer.add_scalar('test/loss', losses.value_avg, epoch)
    
    return test_results





                ###################################### main ######################################
                ###################################### main ######################################
                ###################################### main ######################################


if __name__ == '__main__':
    

    log_path = "./log/"
    os.makedirs(log_path, exist_ok=True)

    model_name = 'TEXT'
    sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
    print(f"Log file name (modelname): {model_name}.log")



    dataPath = {
            'mosi': './datasets/mosi_unaligned_50_reason.pkl',
            'sims': './datasets/sims_unaligned_39_reason.pkl', # You can comment out datasets you don't need
            'sims2': './datasets/sims2_unaligned_50_reason.pkl',
            'mosei': './datasets/mosei_unaligned_50_reason.pkl',
    }

    SMoE_layers = {
        'mosi': 3,
        'sims': 3,
        'sims2': 1,
        'mosei': 2,
    }


    for data in dataPath.keys():

        for i in range(3):

            opt = parse_opts()
            print("############################################################################### {} ###############################################################################".format(data))
            opt.use_bert = True
            opt.need_truncated = True
            opt.need_data_aligned = True
            opt.need_normalize = False
            opt.datasetName = data
            opt.dataPath = dataPath[data]
            opt.CUDA_VISIBLE_DEVICES = opt.device
            opt.is_test = 1
            opt.batch_size = 64
            opt.result_txtPath = "./results.txt"
            opt.fusion_layer_depth = SMoE_layers[data]

            if data == 'mosei': # 'mosei' easily overfit cause of larger dataset 
                opt.epochs = 101

            if opt.text_bert == 'text_bert' and 'sims' in data:
                opt.text_bert = 'text_en_bert'

            opt.seq_lens = [50, 50, 50]

            device = torch.device('cuda:{}'.format(opt.CUDA_VISIBLE_DEVICES) if torch.cuda.is_available() else 'cpu')

            main(opt, device)





            bast_results_base_val = best_result_val(val_results_list, test_results_list)

            # with open(opt.result_txtPath, 'a') as f:
            #     f.write("Metrrr:   ")
            #     for key, (value, epoch) in bast_results_base_val.items():
            #         f.write(f"{value:<5.4f} -- {epoch:<5}")
            #     f.write(f"{opt.datasetName:<8}【{opt.note}】")
            #     f.write("\n")

            
            # with open(opt.result_txtPath, 'a') as f:
            #     f.write("Resuuu:   ")
            #     for key, value in val_global_acc2_best['test_result'].items():
            #         f.write(f"{value:<15.4f}")
            #     f.write(f"{opt.datasetName:<8}【{opt.note}】{val_global_acc2_best['epoch']}\n\n")

            val_results_list = []
            test_results_list = []





        with open(opt.result_txtPath, 'a') as f:
            f.write("########################################## {} \n".format(data))
            f.write("Metric:   ")
            for key in bast_results_base_val.keys():
                f.write(f"{key:<15}")
            f.write("\n")


        with open(opt.result_txtPath, 'a') as f:
            f.write("Result:   ")
            for key, value in val_global_acc2_best['test_result'].items():
                f.write(f"{value:<15.4f}")
            f.write(f"{opt.datasetName:<8}【{opt.note}】{val_global_acc2_best['epoch']}\n\n\n\n\n")

        
        print(val_global_acc2_best['test_result'], '\n\n\n\n\n\n')


        val_global_acc2_best = {'epoch': 0, 'val_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}, 'test_result': {'Has0_acc_2': 0.0, 'Mult_acc_2': 0.0}}
