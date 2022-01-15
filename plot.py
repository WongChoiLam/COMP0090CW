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

def plot_baseline_COCO_ablation(base_path,COCO_path,ablation_path):
    Baseline_stats  = read(base_path)
    COCO_stats  = read(COCO_path)
    Ablation_stats  = read(ablation_path)
    
    train_loss_Baseline = np.array(Baseline_stats[0],dtype=float)
    valid_loss_Baseline = np.array(Baseline_stats[1],dtype=float)

    train_loss_COCO = np.array(COCO_stats[0],dtype=float)
    valid_loss_COCO = np.array(COCO_stats[1],dtype=float)
    
    train_loss_Ablation = np.array(Ablation_stats[0],dtype=float)
    valid_loss_Ablation = np.array(Ablation_stats[1],dtype=float)

    x = np.linspace(1, len(Baseline_stats[0]), len(Baseline_stats[0]))
    
    plt.figure()
    l1, = plt.plot(x, train_loss_Baseline, color='C1', linewidth=1.0, linestyle='-')
    l2, = plt.plot(x, valid_loss_Baseline, color='C1', linewidth=1.0, linestyle=':')
    l3, = plt.plot(x, train_loss_COCO, color='C2', linewidth=1.0, linestyle='-')
    l4, = plt.plot(x, valid_loss_COCO, color='C2', linewidth=1.0, linestyle=':')
    l5, = plt.plot(x, train_loss_Ablation, color='C5', linewidth=1.0, linestyle='-')
    l6, = plt.plot(x, valid_loss_Ablation, color='C5', linewidth=1.0, linestyle=':')
    
    plt.legend(handles=[l1,l3,l5], 
            labels=['Baseline',
                    'COCO',
                    'Ablation'],  loc='best')

    plt.xlim((0, len(Baseline_stats[0])+1))
    plt.xlabel('Num of epoches')
    plt.ylabel('Loss value')
    plt.savefig('Ablation.png')

def plots_baseline_OEQ(base_path,VOC_path,city_path,ISIC_path,MAS3K_path):
    Baseline_stats  = read(base_path)
    VOC_stats  = read(VOC_path)
    city_stats = read(city_path)
    ISIC_stats = read(ISIC_path)
    MAS3K_stats = read(MAS3K_path)
    
    train_loss_Baseline = np.array(Baseline_stats[0],dtype=float)
    valid_loss_Baseline = np.array(Baseline_stats[1],dtype=float)

    train_loss_VOC = np.array(VOC_stats[0],dtype=float)
    valid_loss_VOC = np.array(VOC_stats[1],dtype=float)

    train_loss_city = np.array(city_stats[0],dtype=float)
    valid_loss_city = np.array(city_stats[1],dtype=float)
    
    train_loss_ISIC = np.array(ISIC_stats[0],dtype=float)
    valid_loss_ISIC = np.array(ISIC_stats[1],dtype=float)
    
    train_loss_MAS3K = np.array(MAS3K_stats[0],dtype=float)
    valid_loss_MAS3K = np.array(MAS3K_stats[1],dtype=float)
    
    x = np.linspace(1, len(Baseline_stats[0]), len(Baseline_stats[0]))
    
    plt.figure()
    l1, = plt.plot(x, train_loss_Baseline, color='C1', linewidth=1.0, linestyle='-')
    l2, = plt.plot(x, valid_loss_Baseline, color='C1', linewidth=1.0, linestyle=':')
    l3, = plt.plot(x, train_loss_VOC, color='C2', linewidth=1.0, linestyle='-')
    l4, = plt.plot(x, valid_loss_VOC, color='C2', linewidth=1.0, linestyle=':')
    l5, = plt.plot(x, train_loss_city, color='C3', linewidth=1.0, linestyle='-')
    l6, = plt.plot(x, valid_loss_city, color='C3', linewidth=1.0, linestyle=':')
    l7, = plt.plot(x, train_loss_ISIC, color='C4', linewidth=1.0, linestyle='-')
    l8, = plt.plot(x, valid_loss_ISIC, color='C4', linewidth=1.0, linestyle=':')
    l9, = plt.plot(x, train_loss_MAS3K, color='C5', linewidth=1.0, linestyle='-')
    l10, = plt.plot(x, valid_loss_MAS3K, color='C5', linewidth=1.0, linestyle=':')
    
    # plt.legend(handles=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10], 
    #         labels=['Baseline: training loss', 'Baseline: validation loss',
    #                 'VOC2012: training loss','VOC2012: validation loss',
    #                 'Cityscapes: training loss','Cityscapes: validation loss',
    #                 'ISIC2018: training loss','ISIC2018: validation loss',
    #                 'MAS3K: training loss','MAS3K: validation loss'], loc='best')

    plt.legend(handles=[l1,l3,l5,l7,l9], 
            labels=['Baseline',
                    'VOC2012',
                    'Cityscapes',
                    'ISIC2018',
                    'MAS3K'], loc='upper right')

    plt.xlim((0, len(Baseline_stats[0])+1))
    plt.xlabel('Num of epoches')
    plt.ylabel('Loss value')
    plt.savefig('OEQ.png')

if __name__ == '__main__':
    plot_baseline_COCO_ablation('BaseLine_stats.csv','COCO_stats.csv','Ablation_stats.csv')    
    plots_baseline_OEQ('BaseLine_stats.csv','VOC_stats.csv','cityscapes_stats.csv','ISIC_stats.csv','MAS3K_stats.csv')