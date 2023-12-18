from easydict import EasyDict

cfg = EasyDict(data={}, train={})

#ImageNet mean and std
cfg.data.data_mean = [0.485, 0.456, 0.406]
cfg.data.data_std  = [0.229, 0.224, 0.225]

cfg.data.dataset_dir = "classification_dataset"

# Define train and val csv paths
cfg.data.train_path = f"{cfg.data.dataset_dir}/train/train.csv"
cfg.data.val_path = f"{cfg.data.dataset_dir}/val/val.csv"

cfg.data.img_size = (320,576)  #(height, width)

cfg.train.pretrain = True


cfg.train.lr = 1e-3
cfg.train.patience =5
cfg.train.lr_factor = 0.55


cfg.train.epochs = 1
cfg.train.start_epoch = 0


cfg.train.weight_decay = 0 # 0


cfg.train.save_interval = 2


#new
cfg.train.uniqueid = "lampv1_18dec"

cfg.train.model_type = "efficientnetb4"

cfg.train.use_wandb = True

cfg.train.use_aug = False

cfg.train.save_model = True

cfg.train.classes = ["cracked", "fadelamp", "foggy"]

cfg.train.batch_size = 16

cfg.train.log_dir = "classification_logs/config_3"
