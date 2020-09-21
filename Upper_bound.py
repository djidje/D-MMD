import torchreid
import torch.nn as nn

source = 'dukemtmcreid'
target = 'msmt17'

batch_size = 32

path_market1501 = 'log/market1501/market1501_to_market1501/model.pth.tar-30'
path_dukemtmcreid = 'log/dukemtmcreid/dukemtmcreid_to_dukemtmcreid/model.pth.tar-30'
path_msmt17 = 'log/msmt17/msmt17_to_msmt17'

datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=target,
        targets=target,
        height=256,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=False
)

model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True
)

model = model.cuda()
#model = nn.DataParallel(model).cuda()

optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=50
)

if target == 'dukemtmcreid':
        torchreid.utils.load_pretrained_weights(model, path_market1501)
else:
        torchreid.utils.load_pretrained_weights(model, path_dukemtmcreid)


#model = model.cuda()
model = nn.DataParallel(model).cuda()

engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
)

engine.run(
        save_dir='log/upper_bound/source_' + source + '_target_' + target,
        max_epoch=150,
        eval_freq=30,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=0
)
