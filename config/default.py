from yacs.config import CfgNode as Node
cfg = Node()
cfg.seed = 324
cfg.dataset = 'california_housing'
cfg.task = 'regression'
cfg.resume_dir = ''

cfg.model = Node()
cfg.model.base_outdim = 64
cfg.model.k = 5
cfg.model.drop_rate = 0.1
cfg.model.layer = 4

cfg.logname = 'layer' + str(cfg.model.layer) #+ '_' + str(cfg.model.base_outdim)


cfg.fit = Node()
cfg.fit.lr = 0.008
cfg.fit.max_epochs =2000
cfg.fit.patience = 100
cfg.fit.batch_size = 8192
cfg.fit.virtual_batch_size = 256
cfg.fit.schedule_step = 30