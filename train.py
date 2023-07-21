import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models

from utils import ImageTransform, MyDataset
from fix_seed import fix_seed
from save_results import save_confusionmatrix
from get_files import get_files_list

USE_FINE_TUNING = False

def validation(net, device, criterion, val_dataloader):
    net.eval()
    total_loss = 0
    Y = []
    preds = []
    with tqdm(total=len(val_dataloader)) as pbar:
        pbar.set_description('validation')
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1) # ラベルの予想
            total_loss += loss.item() * inputs.size(0)
            Y.extend(labels)
            preds.extend(pred)
            pbar.update(1)
    return total_loss, Y, preds

def train():
    random_seed = 1234 # 乱数シード
    mode = 'toyota_cars'
    path_input = os.path.join('dataset_' + mode)
    model = 'VGG16'
    use_pretrained = True
    batch_size = 32 # ミニバッチサイズ
    lr = 0.001 # 学習率
    max_itr = 100 # 最大イテレーション数
    val_interval = 10 # validation間隔
    dir_output = os.path.join('output') # output dir

    fix_seed(random_seed) # fix random seed

    # ログ用ファイルの用意
    output_save_path = os.path.join(dir_output, 'itr{0}_{1}pre:{2}_{3}'.format(
        max_itr,
        model,
        use_pretrained,
        path_input
    ))
    os.makedirs(output_save_path, exist_ok=True)

    # 学習の条件を記録
    with open(os.path.join(output_save_path, 'conditions.txt'), 'w') as logfile:
        logfile.write('random_seed:{}\n'.format(random_seed))
        logfile.write('path_input:{}\n'.format(path_input))
        logfile.write('model:{}\n'.format(model))
        logfile.write('use_pretrain:{}\n'.format(use_pretrained))
        logfile.write('USE_FINETUNING:{}\n'.format(USE_FINE_TUNING))
        logfile.write('batch_size:{}\n'.format(batch_size))
        logfile.write('lr:{}\n'.format(lr))
        logfile.write('max_itr:{}\n'.format(max_itr))
        logfile.write('val_interval:{}\n'.format(val_interval))
        logfile.write('dir_output:{}\n'.format(dir_output))

    # 学習に使うデータをリストでまとめる
    list_file = get_files_list(path_input=path_input, mode=mode)

    # データをtrain, val, testの3つに分割
    list_train, list_val = train_test_split(list_file, shuffle=True, random_state=random_seed, test_size=0.2)
    list_val, list_inference = train_test_split(list_val, shuffle=True, random_state=random_seed, test_size=0.5)

    # ImageNetでpretrainedのVGG16を使用する
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)

    # dataset
    train_dataset = MyDataset(list_train, transform=transform, phase='train')
    val_dataset = MyDataset(list_val, transform=transform, phase='val')

    # dataloader
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # デバイスを選択
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ネットワークの選択
    if model == 'VGG16':
        # net = models.vgg16(pretrained=use_pretrained)
        net = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        # 出力層を2つに付け替える
        net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    else:
        print('error; unsuitable network')
        return

    # 訓練モードに
    net.train()
    # デバイスへ転送
    net.to(device)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の選択
    if USE_FINE_TUNING:
        # 最適化手法を設定
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        # 転移学習
        params_to_update = []
        update_param_names = ['classifier.6.weight', 'classifier.6.bias']
        for name, param in net.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        optimizer = optim.Adam(params=params_to_update, lr=lr)

    path_save_logfile_train = os.path.join(output_save_path, 'log_score_train.csv')
    path_save_logfile_val = os.path.join(output_save_path, 'log_score_val.csv')
    with open(path_save_logfile_train, 'w') as logfile:
        logfile.write('iteration,time,loss,accuracy,precision,recall\n')
    with open(path_save_logfile_val, 'w') as logfile:
        logfile.write('iteration,time,loss,accuracy,precision,recall\n')

    # 学習
    ## 最初にvalidation
    time_start = time.perf_counter()
    val_loss, Y, preds = validation(net=net, device=device, criterion=criterion, val_dataloader=val_dataloader)
    accuracy = accuracy_score(y_true=Y, y_pred=preds)
    precision = precision_score(y_true=Y, y_pred=preds)
    recall = recall_score(y_true=Y, y_pred=preds)
    time_end = time.perf_counter()
    with open(path_save_logfile_val, 'a') as logfile:
        logfile.write('0,{},{},{},{},{}\n'.format(time_end-time_start, val_loss, accuracy, precision, recall))
    save_confusionmatrix(y_true=Y, y_pred=preds, path_save=output_save_path, phase='initial_validation')

    count = 0
    iteration = 0
    total_loss = 0
    Y_train = []
    pred_train = []
    time_trainval_total_start = time.perf_counter()
    with tqdm(total=max_itr) as pbar:
        pbar.set_description('training')
        while iteration < max_itr:
            for inputs, labels in train_dataloader:
                if iteration >= max_itr:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)
                if count == 0:
                    net.train()
                    time_trainval_interval_start = time.perf_counter()
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, pred = torch.max(outputs, 1) # ラベルの予測

                    # バックプロパゲーション
                    loss.backward()
                    optimizer.step()

                # カウント
                count += 1
                iteration += 1
                # 損失計算
                total_loss += loss.item() * inputs.size(0)
                # スコア計算用
                Y_train.extend(labels)
                pred_train.extend(pred)

                if count == val_interval:
                    time_trainval_interval_end = time.perf_counter()
                    time_trainval_interval = time_trainval_interval_end - time_trainval_interval_start
                    total_loss = total_loss / val_interval
                    # validation
                    loss_val, Y, preds = validation(net, device=device, criterion=criterion, val_dataloader=val_dataloader)
                    # get scores
                    ## train
                    accuracy_train = accuracy_score(y_true=Y_train, y_pred=pred_train)
                    precision_train = precision_score(y_true=Y_train, y_pred=pred_train)
                    recall_train = recall_score(y_true=Y_train, y_pred=pred_train)
                    ## validation
                    accuracy_val = accuracy_score(y_true=Y, y_pred=preds)
                    precision_val = precision_score(y_true=Y, y_pred=preds)
                    recall_val = recall_score(y_true=Y, y_pred=preds)
                    # save log
                    ## training
                    with open(path_save_logfile_train, 'a') as logfile:
                        logfile.write('{},{},{},{},{},{}\n'.format(iteration, time_trainval_interval, total_loss, accuracy_train, precision_train, recall_train))
                    ## validation
                    with open(path_save_logfile_val, 'a') as logfile:
                        logfile.write('{},{},{},{},{},{}\n'.format(iteration, time_trainval_interval, loss_val, accuracy_val, precision_val, recall_val))

                    # 結果の描画

                    # reset
                    count = 0
                    Y_train = []
                    pred_train = []

                pbar.update(1)

    time_trainval_total_end = time.perf_counter()
    with open(os.path.join(output_save_path, 'logtime.txt'), 'w') as logfile:
        logfile.write('total time: {}\n'.format(time_trainval_total_end - time_trainval_total_start))

    # ネットワーク重みの保存
    save_path = os.path.join(output_save_path, 'weight.pth')
    torch.save(net.state_dict(), save_path)
    # 混合行列を保存
    save_confusionmatrix(y_true=Y, y_pred=preds, path_save=output_save_path, phase='validation')

    # inferenceに引き継ぐパラメータを保存
    dict_train = {
        'random_seed':random_seed,
        'path_input':path_input,
        'batch_size':batch_size,
        'output_save_path':output_save_path,
        'list_inference':list_inference,
        'transform': transform,
        'net':net,
        'criterion':criterion
    }

    return dict_train

def inference(dict_train):
    # dict_trainからパラメータを抽出
    random_seed = dict_train['random_seed']
    path_input = dict_train['path_input']
    batch_size = dict_train['batch_size']
    output_save_path = dict_train['output_save_path']
    list_inference = dict_train['list_inference']
    transform = dict_train['transform']
    net = dict_train['net']
    criterion = dict_train['criterion']

    fix_seed(random_seed) # fix random seed

    # dataset
    inference_dataset = MyDataset(list_inference, path_input, transform=transform, phase='val')

    # dataloader
    inference_dataloader = data.DataLoader(inference_dataset, batch_size=batch_size, shuffle=True)

    # デバイスを選択
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.eval()
    net.to(device)

    # 重みのロード
    # path_weight = os.path.join(output_save_path, 'weight.pth')
    # load_weights = torch.load(path_weight)
    # net.load_state_dict(load_weights)

    total_loss = 0
    Y = []
    preds = []

    with tqdm(total=len(inference_dataloader)) as pbar:
        pbar.set_description('Inference')
        for inputs, labels in inference_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            total_loss += loss.item()
            Y.extend(labels)
            preds.extend(pred)
            pbar.update(1)

    # 結果の保存
    accuracy = accuracy_score(y_true=Y, y_pred=preds)
    precision = precision_score(y_true=Y, y_pred=preds)
    recall = recall_score(y_true=Y, y_pred=preds)
    path_save_result = os.path.join(output_save_path, 'result_inference.csv')
    with open(path_save_result, 'w') as logfile:
        logfile.write('loss,accuracy,precision,recall\n')
        logfile.write('{},{}\n'.format(total_loss, accuracy, precision, recall))

    # 混同行列の保存
    save_confusionmatrix(y_true=Y, y_pred=preds, path_save=output_save_path, phase='inference')

if __name__=='__main__':
    # 学習と検証
    dict_train = train()
    # テストデータでの検証
    inference(dict_train)
