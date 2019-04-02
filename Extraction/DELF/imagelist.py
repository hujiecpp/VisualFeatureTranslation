# from PIL import Image
import os
import natsort
import argparse
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", "--dir", required=True, help="data dir")
args = vars(ap.parse_args())
path=args["dir"]
files = natsort.natsorted(os.listdir(path))
f=open('./list_images.txt','w')
i=0
for file in files:
    f.write(path+'/'+file+'\n')
   # try:
   #     im=Image.open(path+'/'+file)
   #     f.write(file+'\n')
   #     im=im.convert('RGB')
   #     im.save(path+'_bmp/'+(file.replace('.JPEG','.bmp')))
   #     i=i+1
   #     print(i)
   # except:
   #     i=i+1
   #     print('error:',i)
   #     continue;
f.close()
