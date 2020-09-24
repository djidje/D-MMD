import torchreid

source = 'dukemtmcreid'
target = 'market1501'

batch_size = 32

datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=source,
        targets=target,
        height=256,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erasing'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True
)

model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='mmd',
        pretrained=True
)

model = model.cuda()
#model = nn.DataParallel(model).cuda() # Comment previous line and uncomment this line for multi-gpu use

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

# We use pretrained model to continue the training on the target domain
start_epoch = torchreid.utils.resume_from_checkpoint(
        'log/source_training/' + source + '/model/model.pth.tar-30',
        model,
        optimizer
)

model = model.cuda()
#model = nn.DataParallel(model).cuda()  # Comment previous line and uncomment this line for multi-gpu use

engine = torchreid.engine.ImageMmdEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        mmd_only=False,
)

# Define the lower bound - test without adaptation
engine.run(
        test_only=True,
)

# Start the domain adaptation
engine.run(
        save_dir='log/first_try_DMMD',
        max_epoch=150,
        eval_freq=5,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=start_epoch
)
