import torchreid

source = 'dukemtmcreid'
target = source

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
        load_train_targets=False
)

model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True
)

model = model.cuda()
#model = nn.DataParallel(model).cuda()  # Comment previous line and uncomment this line for multi-gpu use

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


engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
)

engine.run(
        save_dir='log2/upper_bound/source_' + source + '_target_' + target,
        max_epoch=30,
        eval_freq=10,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=0
)
