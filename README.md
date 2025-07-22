# Project11: Uncertainty - Aware Road Obstacle Identification
A pipeline for semantic segmentation of unexpected obstacles on the road using the [Lost and Found](https://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) dataset and DeepLabV3+.
## Dataset
- Download from: https://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html the zip folders gtCoarse.zip and leftImg8bit.zip
- Apply preprocessing by executing the file preprocessing.py, but be careful substitute the ROOT path with the one where you downloaded the exported dataset.
- Excecute the program in order to do the training by this command: python main.py --dataset lostandfound --data_root ./dataset/lostandfound --model deeplabv3plus_resnet101 --output_stride 16 --lr 0.0001 --total_itrs 200 --batch_size 4 --crop_size 768 --val_interval 100 --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --loss_type boundary 
- Excecute this command for the test: python main.py --dataset lostandfound --data_root ./dataset/lostandfound --model deeplabv3plus_resnet101 --output_stride 16 --batch_size 1 --val_batch_size 1 --test_only --ckpt checkpoints/best_deeplabv3plus_resnet101_lostandfound_os16.pth --loss_type boundary
## Checkpoints
You can find the needed checkpoints at this link: https://drive.google.com/drive/folders/1sbtpxwX9yqNjTYMGUIvyFyxBZ7_Co_AP?usp=drive_link
Download the checkpoints and put them in the checkpoints folder.
## Modified files
- main.py
- preprocessing.py
- datasets/lostandfound.py
## Important
In the main branch there are README file and the presentation, instead the code is in the master branch.
