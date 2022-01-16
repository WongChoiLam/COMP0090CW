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

def plot_loss(base_path,VOC_path,city_path,ISIC_path,MAS3K_path, COCO_path, Ablation_path):
    Baseline_stats  = read(base_path)
    VOC_stats  = read(VOC_path)
    city_stats = read(city_path)
    ISIC_stats = read(ISIC_path)
    MAS3K_stats = read(MAS3K_path)
    COCO_stats = read(COCO_path)
    Ablation_stats  = read(Ablation_path)

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

    train_loss_COCO = np.array(COCO_stats[0],dtype=float)
    valid_loss_COCO = np.array(COCO_stats[1],dtype=float)

    train_loss_Ablation = np.array(Ablation_stats[0],dtype=float)
    valid_loss_Ablation = np.array(Ablation_stats[1],dtype=float)

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
    l11, = plt.plot(x, train_loss_COCO, color='C6', linewidth=1.0, linestyle='-')
    l12, = plt.plot(x, valid_loss_COCO, color='C6', linewidth=1.0, linestyle=':')
    l13, = plt.plot(x, train_loss_Ablation, color='C7', linewidth=1.0, linestyle='-')
    l14, = plt.plot(x, valid_loss_Ablation, color='C7', linewidth=1.0, linestyle=':')   

    plt.legend(handles=[l1,l11,l13,l7,l5,l9,l3], 
            labels=['Baseline',
                    'COCO',
                    'COCO-Ablation',
                    'ISIC2018',
                    'Cityscapes',
                    'MAS3K',
                    'VOC2012'], loc='upper right', bbox_to_anchor=(1.35,1))

    plt.xlim((0, len(Baseline_stats[0])+1))
    plt.xlabel('Num of epoches')
    plt.ylabel('Loss value')
    plt.savefig('Loss.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_loss('BaseLine_stats.csv','VOC_stats.csv','cityscapes_stats.csv','ISIC_stats.csv','MAS3K_stats.csv','COCO_stats.csv','Ablation_stats.csv')