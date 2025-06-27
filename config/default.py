from yacs.config import CfgNode as Node
cfg = Node()
cfg.seed = 324
cfg.dataset = 'california_housing'
cfg.task = 'regression'
cfg.resume_dir = ''

cfg.model = Node()
cfg.model.base_outdim = 64
cfg.model.k = 5
cfg.model.drop_rate = 0.2
cfg.model.layer = 2

cfg.logname = 'BATCHES_PROOFS' + str(cfg.model.layer) #+ '_' + str(cfg.model.base_outdim)


cfg.fit = Node()
cfg.fit.lr = 0.001
cfg.fit.max_epochs =200
cfg.fit.patience = 10
cfg.fit.batch_size = 64
cfg.fit.virtual_batch_size = 128
cfg.fit.schedule_step = 10