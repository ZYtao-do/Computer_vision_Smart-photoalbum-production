# coding:utf-8
from pathlib import Path
import os
import os.path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import random
# 保持监听的文件
folder_to_track = "./dataset"
# 图片文件

animals = "./data/picture/animal"
food = "./data/picture/food"
people = "./data/picture/human"
scenery = "./data/picture/view"
text = "./data/picture/playbill"
'''
# 音乐文件
folder_destination_music = "./data/music"
# 视频文件
folder_destination_video = "./data/video"
'''

class Hearing(FileSystemEventHandler):
    def on_any_event(self, event):
        for filename in os.listdir(folder_to_track) :
            src = folder_to_track +"/"+ filename
            if filename.endswith("jpg") or filename.endswith("png"):
                #new_destination=folder_destination_picture + "/"+filename
                print("\n查询到新图片：", filename)
                os.system(r"python 3.py -f " + "./dataset/" + filename + " > ./result.txt")
                with open('result.txt', 'r', encoding='GB2312') as f:  # 打开文件
                    lines = f.readlines()  # 读取所有行
                    last_line = lines[-1]  # 取最后一行
                    result = int(last_line)
                    if result == 0:
                        a = "animal"
                    elif result == 1:
                        a = "food"
                    elif result == 2:
                        a = "human"
                    elif result == 3:
                        a = "view"
                    elif result == 4:
                        a = "playbill"
                    print("预测的最终类别为:", a)
                name = random.randint(0, 100000)
                if result==0:
                    new_destination=animals + "/" +filename
                    my_file = Path(new_destination)
                    while(my_file.exists()):
                        new_destination = animals + "/" + str(name) + filename
                        break
                elif result==2:
                    new_destination=people + "/" +filename
                    my_file = Path(new_destination)
                    while(my_file.exists()):
                        new_destination = animals + "/" + str(name) + filename
                        break
                elif result==1:
                    new_destination=food + "/" +filename
                    my_file = Path(new_destination)
                    while(my_file.exists()):
                        new_destination = animals + "/" + str(name) + filename
                        break
                elif result==3:
                    new_destination=scenery + "/" +filename
                    my_file = Path(new_destination)
                    while(my_file.exists()):
                        new_destination = animals + "/" + str(name) + filename
                        break
                elif result==4:
                    new_destination=text + "/" +filename
                    my_file = Path(new_destination)
                    while(my_file.exists()):
                        new_destination = animals + "/" + str(name) + filename
                        break
                    
            elif filename.endswith("mp3") :
                new_destination=folder_destination_music + "/" + filename
            elif filename.endswith("avi") or filename.endswith("mp4"):
                new_destination = folder_destination_video + "/"+filename
            
            
            os.rename(src, new_destination)
            print("已完成相片整理操作")
            break


event_handler = Hearing()
observer = Observer()
observer.schedule(event_handler, folder_to_track, recursive=True)
observer.start()
try:
    while True:
        time.sleep(10)
        # print('正在监听...')
except KeyboardInterrupt:
    observer.stop()

observer.join()