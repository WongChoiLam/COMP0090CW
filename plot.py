import csv
import matplotlib.pyplot as plt
import numpy as np

def read(filepath):  
    data = []
    with open(filepath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    return data

def plot_baseline_COCO(base_path,COCO_path):
    Baseline_stats  = read(base_path)
    COCO_stats  = read(COCO_path)

    train_loss_Baseline = np.array(Baseline_stats[0],dtype=float)
    valid_loss_Baseline = np.array(Baseline_stats[1],dtype=float)

    train_loss_COCO = np.array(COCO_stats[0],dtype=float)
    valid_loss_COCO = np.array(COCO_stats[1],dtype=float)

    x = np.linspace(1, len(Baseline_stats[0]), len(Baseline_stats[0]))
    
    plt.figure()
    l1, = plt.plot(x, train_loss_Baseline, color='C1', linewidth=1.0, linestyle='-')
    l2, = plt.plot(x, valid_loss_Baseline, color='C1', linewidth=1.0, linestyle=':')
    l3, = plt.plot(x, train_loss_COCO, color='C2', linewidth=1.0, linestyle='-')
    l4, = plt.plot(x, valid_loss_COCO, color='C2', linewidth=1.0, linestyle=':')

    plt.legend(handles=[l1,l2,l3,l4], 
            labels=['Baseline: training loss', 'Baseline: validation loss',
                    'COCO transfer: training loss','COCO transfer: validation loss'],  loc='best')

    plt.xlim((0, len(Baseline_stats[0])+1))

    plt.xlabel('num of epoches')
    plt.ylabel('loss value')
    plt.show()

def plots_ablation_study(file_path):
    Ablation_stats  = read(file_path)
    
    train_loss_Ablation_freeze = np.array(Ablation_stats[0],dtype=float)
    valid_loss_Ablation_freeze = np.array(Ablation_stats[1],dtype=float)
    train_loss_Ablation_unfreeze = np.array(Ablation_stats[2],dtype=float)
    valid_loss_Ablation_unfreeze = np.array(Ablation_stats[3],dtype=float)
    train_loss_Ablation = np.array(Ablation_stats[4],dtype=float)
    valid_loss_Ablation = np.array(Ablation_stats[5],dtype=float)
    
    x = np.linspace(1, len(Ablation_stats[0]), len(Ablation_stats[0]))
    
    l1, = plt.plot(x, train_loss_Ablation_freeze, color='C3', linewidth=1.0, linestyle='-')
    l2, = plt.plot(x, valid_loss_Ablation_freeze, color='C3', linewidth=1.0, linestyle=':')
    l3, = plt.plot(x, train_loss_Ablation_unfreeze, color='C4', linewidth=1.0, linestyle='-')
    l4, = plt.plot(x, valid_loss_Ablation_unfreeze, color='C4', linewidth=1.0, linestyle=':')
    l5, = plt.plot(x, train_loss_Ablation, color='C5', linewidth=1.0, linestyle='-')
    l6, = plt.plot(x, valid_loss_Ablation, color='C5', linewidth=1.0, linestyle=':')
    
    plt.legend(handles=[l1,l2,l3,l4,l5,l6], 
           labels=['Ablation: training loss freeezed version','Ablation: validation loss freezed version',
                   'Ablation: training loss unfreeezed version','Ablation: validation loss unfreeezed version',
                   'Ablation: training loss','Ablation: validation loss'],  loc='best')
    
    plt.xlim((0, len(Ablation_stats[0])+1))
    plt.xlabel('num of epoches')
    plt.ylabel('loss value')
    plt.show()
    
    
plot_baseline_COCO('Baseline_stats.csv','COCO_stats.csv')    
plots_ablation_study('Ablation_stats.csv')

