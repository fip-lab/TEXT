import torch
from opts import *
from core.dataset import MMDataset
from core.utils import AverageMeter, best_result, setup_seed
from model.TEXT import TEXT
from core.metric import MetricsTop

import sys
sys.argv = ['']


dataPath = {
        'mosi': './datasets/mosi_unaligned_50_reason.pkl',
        'sims': './datasets/sims_unaligned_39_reason.pkl',
        'sims2': './datasets/sims2_unaligned_50_reason.pkl',
        'mosei': './datasets/mosei_unaligned_50_reason.pkl',
}

SMoE_layers = {
        'mosi': 3,
        'sims': 3,
        'sims2': 1,
        'mosei': 2,
}

checjpointPath = {
        'mosi': './checkpoint/test/mosi/epoch_79_3layers.pth',
        'sims': './checkpoint/test/sims/epoch_20_3layers.pth',
        'sims2': './checkpoint/test/sims2/epoch_28_1layers.pth',
        'mosei': './checkpoint/test/mosei/epoch_4_2layers.pth',
}


test_results_list = []

for data in dataPath.keys():
    opt = parse_opts()
    opt.device = 0
    print("################################################################ {} ################################################################".format(data))
    opt.use_bert = True
    opt.need_truncated = True
    opt.need_data_aligned = True
    opt.need_normalize = False
    opt.datasetName = data
    opt.dataPath = dataPath[data]
    opt.CUDA_VISIBLE_DEVICES = opt.device
    opt.seq_lens = [50, 50, 50]
    opt.is_test = 1
    opt.batch_size = 1

    opt.fusion_layer_depth = SMoE_layers[data]
    opt.text_bert = 'text_residual_bert'
    opt.test_checkpoint = checjpointPath[data]


    device = torch.device('cuda:{}'.format(opt.device))
    print(device)

    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))

    model = TEXT(dataset=opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth).to(device)
    checkpoint = torch.load(opt.test_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # print(model)

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    dataset = MMDataset(opt, mode='test')


    test_pbar = enumerate(dataset)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:

            text, audio, img, audio_clue, visual_clue = data['text'].to(device), data['audio'].to(device), data['vision'].to(device), data['audio_clue_bert'].to(device), data['visual_clue_bert'].to(device)

            text, audio, img, audio_clue, visual_clue = text.unsqueeze(0), audio.unsqueeze(0), img.unsqueeze(0), audio_clue.unsqueeze(0), visual_clue.unsqueeze(0)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]


            output = model(text=text, audio=audio, visual=img, audio_clue=audio_clue, visual_clue=visual_clue)
            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())


        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        test_results_list.append(test_results)



    print("\n =================== print result =================== \n")
    best_results = best_result(test_results_list)

    print("指标：   ", end="")
    for key in best_results.keys():
        print(f"{key:<15}", end="")
    print()

    print("val_best_acc2:    ", end="")
    for key, (value, epoch) in best_results.items():
        print(f"{value:<15.4f}", end="")
    print(f"{opt.datasetName}  -- {opt.note}")
    print(f"\n\n\n")

    test_results_list = []
