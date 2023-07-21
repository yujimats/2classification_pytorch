import os
import pandas as pd
import matplotlib.pyplot as plt

def save_score_vs_itr(path_output):
    list_phase = ['train', 'val']
    for phase in list_phase:
        path_log_file = os.path.join(path_output, f'log_score_{phase}.csv')
        df = pd.read_csv(path_log_file)
        # 描画
        fig = plt.figure(figsize=(10,8))
        ## loss
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(df['iteration'], df['loss'])
        ax1.set_yscale('log')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('loss')
        ## acc
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.plot(df['iteration'], df['accuracy'])
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('accuracy')
        ## precision
        ax1 = fig.add_subplot(2, 2, 3)
        ax1.plot(df['iteration'], df['precision'])
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('precision')
        ## precision
        ax1 = fig.add_subplot(2, 2, 4)
        ax1.plot(df['iteration'], df['recall'])
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('recall')

        # save and release memory
        plt.savefig(os.path.join(path_output, f'score_vs_itr_{phase}.png'))
        plt.close()

    # train
    path_log_file = os.path.join(path_output, 'log_score_train.csv')

    return

if __name__=='__main__':
    save_score_vs_itr(path_output=os.path.join('output', 'itr200_VGG16pre:True_dataset_toyota_cars'))
