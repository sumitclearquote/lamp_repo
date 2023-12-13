from easydict import EasyDict

cfg = EasyDict(data={}, train={})

#ImageNet mean and std
cfg.data.data_mean = [0.485, 0.456, 0.406]
cfg.data.data_std  = [0.229, 0.224, 0.225]

cfg.data.dataset_dir = "classification_dataset"


cfg.data.img_size = (224,224)  #(height, width)

cfg.train.pretrain = True
cfg.train.lr = 1e-4

cfg.train.epochs = 40
cfg.train.start_epoch = 0

cfg.train.weight_decay = 0

cfg.train.store_results = True

cfg.train.save_interval = 1

#cfg.train.save_model_dir = "classification_logs/config_1"


#new
cfg.train.uniqueid = "lampv1_13dec"

cfg.train.model_type = "efficientnetb4"

cfg.train.use_wandb = False

cfg.train.use_aug = True

cfg.train.save_model = True

cfg.train.classes = ["cracked", "fadelamp", "foggy"]

cfg.train.batch_size = 16

cfg.train.log_dir = "classification_logs/config_1"