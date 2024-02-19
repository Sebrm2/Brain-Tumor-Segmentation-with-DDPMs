import csv
import matplotlib.pyplot as plt
import numpy as np
import os 
def save_value(dice: float, filename: str):
        with open(filename, 'a', newline='') as file:
            file.write(str(dice) + '\n')

def save_image(model, epoch, i, x, imagen_np, masken_np, image_np, mask_np,type):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    bbox = 0, 0, 0, 0
    segmentation = np.where(mask_np == 1)
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, x_max, y_min, y_max
    overlays = np.ma.masked_where(masken_np == 0, masken_np)
    axs[0].imshow(np.mean(imagen_np, axis=0), cmap='gray')
    axs[0].imshow(overlays, cmap='jet', alpha=0.5)
    axs[0].set_title('Anotación')
    overlay = np.ma.masked_where(mask_np == 0, mask_np)
    axs[1].imshow(np.mean(image_np, axis=0), cmap='gray')  
    axs[1].imshow(overlay, cmap='jet', alpha=0.5)
    axs[1].set_title('Predicción')           
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle(f'{model}')
    plt.savefig(f'plots/{model}/{epoch}/{type}-{i}-{x}-real-vs-fake.png')
    plt.close()

def plot_image(data_anot,target_anot,data_anot_post,target_anot_post,epoch,model,x):
    directory = os.path.join("plots",model,str(epoch))
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fp= sum('FP' in element for element in os.listdir(directory))
    fn= sum('FN' in element for element in os.listdir(directory))
    tp= sum('TP' in element for element in os.listdir(directory))

    for i in range(len(data_anot)):
        datis, targit = data_anot[i], target_anot[i]
        imagen_np = datis.squeeze().numpy()
        masken_np = targit.squeeze().numpy()
        datito, predi = data_anot_post[i].cpu().numpy(), target_anot_post[i]
        image_np = datito.squeeze()
        mask_np = predi.squeeze()
        if np.any(mask_np) and np.any(masken_np) and tp <10:
            save_image(model, epoch, i, x, imagen_np, masken_np, image_np, mask_np,'TP')
        if np.any(mask_np) and not np.any(masken_np) and fp <5:
            save_image(model, epoch, i, x, imagen_np, masken_np, image_np, mask_np,'FP')
        if not np.any(mask_np) and np.any(masken_np) and fn <5:
            save_image(model, epoch, i, x, imagen_np, masken_np, image_np, mask_np,'FN')
        

class Graphics(object):
    def __init__(self, input_filename,output_filename,xlabel,ylabel,title,type) -> None:
        self.x = np.array([], dtype=float)
        self.y = []
        self.inputFilename = input_filename
        self.outputFilename = output_filename
        self.xLabel = xlabel
        self.yLabel = ylabel
        self.Title = title
        self.Type = type

    def plot(self):
        with open(self.inputFilename, 'r') as file:
            for line in file:
                self.x= np.append(self.x,float(line.strip()))
        lista = [f"{i}" for i in range(1, len(self.x)+1)]
        self.y = np.array(lista, dtype=int)
        
        
        if self.Type == "bar":
            fig, ax = plt.subplots()
            bar_height = 0.8
            bar_width = 0.4
            left = np.arange(len(self.x))
            rects = ax.barh(left, self.x, height=bar_height, color='b', edgecolor='black', linewidth=1.2, alpha=0.8)
            ax.set_xlabel(self.xLabel)
            ax.set_ylabel(self.yLabel)
            ax.set_title(self.Title)
            ax.set_yticks(left + bar_width / 2)
            ax.set_yticklabels(left + 1)
            ax.tick_params(axis='y', labelsize=8) 
            plt.tight_layout()
            plt.savefig(f"{self.outputFilename}.png")
            
            
        if self.Type == "line":
            fig, ax = plt.subplots()
            ax.set_ylabel(self.yLabel)
            ax.set_xlabel(self.xLabel)
            ax.plot(self.y, self.x, 'o-')
            ax.set_title(self.Title)
            plt.savefig(f"{self.outputFilename}.png")