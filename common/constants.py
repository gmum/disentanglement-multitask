# Strings
LOSS = 'loss'
ACCURACY = 'acc'
ITERATION = 'iteration'
WANDB_NAME = 'disentanglement'
INPUT_IMAGE = 'input_image'
RECON_IMAGE = 'recon_image'
RECON = 'recon'
FIXED = 'fixed'
SQUARE = 'square'
ELLIPSE = 'ellipse'
HEART = 'heart'
TRAVERSE = 'traverse'
RANDOM = 'random'
TEMP = 'tmp'
GIF = 'gif'
JPG = 'jpg'
FACTORVAE = 'FactorVAE'
DIPVAEI = 'DIPVAEI'
DIPVAEII = 'DIPVAEII'
BetaTCVAE = 'BetaTCVAE'
INFOVAE = 'InfoVAE'
TOTAL_VAE = 'total_vae'
TOTAL_VAE_EPOCH = 'total_vae_epoch'
LEARNING_RATE = 'learning_rate'
DSPRITES_NUM_CLASSES1 = 3
DSPRITES_NUM_CLASSES2 = 6
MULTITASKDATA = ('None', 'dsprites_full')
CELEBA_TRAIN_SIZE = 182637
CELEB_NUM_CLASSES = 10
CELEB_CLASSES = (5, 15, 16, 18, 20, 22, 26, 31, 35, 39)

# Algorithms
ALGS = ('CelebAMT', 'DspritesMT', 'FullRegression')

LOSS_TERMS = (FACTORVAE, DIPVAEI, DIPVAEII, BetaTCVAE, INFOVAE)

# Datasets
DATASETS = ('celebA', 'dsprites_full', 'dsprites_multitask', 'dsprites_noshape', 'color_dsprites', 'noisy_dsprites',
            'scream_dsprites',
            'smallnorb', 'cars3d', 'shapes3d', 'shapes3d_multitask',
            'mpi3d_toy', 'mpi3d_realistic', 'mpi3d_real', 'mpi3d_multitask')
DEFAULT_DATASET = DATASETS[-2]  # mpi3d_realistic
TEST_DATASETS = DATASETS[0:2]  # celebA, dsprites_full

# Architectures
DISCRIMINATORS = ('SimpleDiscriminator', 'SimpleDiscriminatorConv64')
TILERS = ('MultiTo2DChannel',)
DECODERS = ('SimpleConv64', 'ShallowLinear', 'DeepLinear')
ENCODERS = ('SimpleConv64', 'SimpleGaussianConv64', 'PadlessConv64', 'PadlessGaussianConv64',
            'ShallowGaussianLinear', 'DeepGaussianLinear', 'DeepLinear')
HEADERS = ('Header', 'DatasetHeader')

# Evaluation Metrics
EVALUATION_METRICS = ('dci', 'factor_vae_metric', 'sap_score', 'mig', 'irs', 'beta_vae_sklearn', 'unsupervised',
                      'max_corr')

# Schedulers
LR_SCHEDULERS = ('ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                 'CosineAnnealingLR', 'CyclicLR', 'LambdaLR')
SCHEDULERS = ('LinearScheduler',)
