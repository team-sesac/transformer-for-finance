import os, sys
sys.path.append(os.getcwd()) # vscode
from src.ch5_cal.utils import *




if __name__ == '__main__':
    total_return_without_ratio, total_return_with_ratio = calculate_etf_returns(4)
    print(total_return_without_ratio, total_return_with_ratio)