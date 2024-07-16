import os
import glob

def delete_txt_files_in_directory(directory):  
    # 使用 glob 模块查找目录下的所有 .txt 文件  
    txt_files = glob.glob(os.path.join(directory, '*.txt'))  
  
    # 遍历找到的 .txt 文件列表，并逐个删除  
    for file in txt_files:  
        try:  
            os.remove(file)  
            print(f"已删除文件: {file}")  
        except OSError as e:  
            print(f"删除文件时出错: {file} - {e.strerror}")  

def delete_pkl_files_in_directory(directory):   
    pkl_files = glob.glob(os.path.join(directory, '*.pkl'))  
    for file in pkl_files:  
        try:  
            os.remove(file)  
            print(f"已删除文件: {file}")  
        except OSError as e:  
            print(f"删除文件时出错: {file} - {e.strerror}") 
            
def delete_npy_files_in_directory(directory):   
    npy_files = glob.glob(os.path.join(directory, '*.npy'))  
    for file in npy_files:  
        try:  
            os.remove(file)  
            print(f"已删除文件: {file}")  
        except OSError as e:  
            print(f"删除文件时出错: {file} - {e.strerror}") 

def clearTrainingDatum():
    delete_npy_files_in_directory("data")
    
def clearModelCheckpoints():
    delete_txt_files_in_directory("log")
    delete_pkl_files_in_directory("log")


if __name__ == "__main__":
    clearTrainingDatum()
    clearModelCheckpoints()
    pass