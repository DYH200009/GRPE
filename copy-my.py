
import os
import shutil

with open(r"C:\Users\10630\source\GaussianRecon\data\geo-gaussian\R1\KeyFrameTrajectory2.txt") as f:
    lines = f.readlines()

for line in lines:
    line = line.split()
    id = int(float(line[0]))

    file_name = os.path.join('data/geo-gaussian/R1/results', f'frame{id:06d}.jpg')
    shutil.copyfile(file_name, os.path.join('data/geo-gaussian/R1/rgb', f'frame{id:06d}.jpg'))
    pass

