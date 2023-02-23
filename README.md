# hackathon_EDISS_2023

Code structure:
 - Implemented approach (Computer vision):
     #### preprocessing.py: Temporal median filter with probabilitsic sampling to create clean city maps
     [Extracted clean city maps can be found in data/gt folder]
     
     #### main.py: Includes all the supporting code to create a temporal redundacny and constraint based interference removal approach.
     [The data folder will have the provided datasets as - data/pa data/ba data/va]
     
     To visualize output at each step: `python main.py 1`
     The output folder also needs to be formatted in the root as root/out/pa root/out/va root/out/ba. All processed outputs are saved as .png in the out folder.
 - Future Work (automated approach): 
     #### dataset_creator.py: Augments the only cloud images manually collected with manually sourced inference masks; as a result training dataset for segmentation models can be created.
     ## U-2-Net: Forked from https://github.com/xuebinqin/U-2-Net.git U2Net is a very powerful image segmentation model; the training script (u2net_train.py) is modified to train on the created dataset. Due to time and computational constraint, the model could not be adequately trained. Before testing it has to be trained for more than a 1000 epoch.
