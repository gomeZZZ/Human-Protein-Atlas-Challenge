import matplotlib.pyplot as plt 
import numpy as np

def plot_histogram(data, bins = 360,xlabel = 'Value'):
    plt.figure(figsize=(10,8))
    plt.hist(data,bins = bins)
    plt.grid(True)
    plt.title('Histogram',fontsize=26)
    plt.xlabel(xlabel,fontsize=24)
    plt.ylabel('Frequency',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()

def print_horizontal_divider():
    print()
    print("#################################################################################################")
    print() 

def print_data_distribution(y, name = 'Data distribution'):
    print_horizontal_divider()
    print("##########   ", name,"   #############")
    print("MEAN=",np.mean(y),"   STD=",np.std(y))
    print("MIN=",np.min(y),"    MAX=",np.max(y))
    print_horizontal_divider()

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Taken from:https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()