from Utils.utils import *

def log(msg,end=None,verbose=True,divider=False):
    if verbose:
        if divider:
            print_horizontal_divider()
        if end==None:
            print(msg)
        else:
            print(msg,end=end)