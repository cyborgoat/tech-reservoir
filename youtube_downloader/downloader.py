import pathlib
from pytube import YouTube 
from tqdm import tqdm, trange
  
# where to save 
SAVE_PATH = pathlib.Path("output") 
  
# link of the video to be downloaded 
# opening the file 
link=open('CMU-CS-11-711.txt','r') 
  
for i in tqdm(link): 
    url =YouTube(i)
    print(url.title)
    video = url.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video.download(SAVE_PATH)
