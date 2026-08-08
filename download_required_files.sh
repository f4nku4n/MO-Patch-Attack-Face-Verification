# dataset
gdown 1sGtnHX3UYkg47lktNmz8dyQ8iu1hx_Ow
unzip lfw_preprocess.zip
rm lfw_preprocess.zip

# mask
gdown 1rUihWMCy-6zrbWSMttgTW1LSm8b4GQ7G
unzip mask.zip
rm mask.zip

# mkdir
mkdir pretrained_model

# pretrained model
wget -O pretrained_model/vggface2.pt https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
wget -O pretrained_model/webface.pt https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt
gdown -O pretrained_model/arcface.pth 1Hmqf25ZIoVLng3wrg0CgTwzA93XQmRCc
gdown -O pretrained_model/cosface.pth 1hlVcK4mc3kGp4FU_slRVJJD-YIcw5lNr