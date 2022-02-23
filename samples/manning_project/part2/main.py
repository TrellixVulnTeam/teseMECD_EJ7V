# Trains a CNN (ResNet-50) to regress image captions embeddings from images
# The CNN is modifed to have the same number of outputs as the embeddings dimensionality (400)
# It is trained with a Cross-Entropy Loss with Sigmoid activations, which sets the regression of each embedding value as an idependent problem
# Under this configuration, the minimum Loss is not 0. But the minimum Loss is still the best solution for the problem
# The training process includes a validation (test the CNN with unseen data during training). But the only information we get form it is the loss values
# Should train until validation loss does not decrese anymore. Then the CNN will start overfitting, and therefore the retrieval performance will decrese

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import dataset
import train
import model
from pylab import zeros, arange, subplots, plt, savefig
from torch.hub import load_state_dict_from_url

# Config
training_id = 'InstaCities1M_XE_ADAM'
dataset_root = '../../../datasets/InstaCities1M/'
split_train = 'train_InstaCities1M.txt'
split_val = 'val_InstaCities1M.txt'

embedding_dimensionality = 400  # Number of CNN outputs (dimensionality of the word2vec model)
batch_size = 64 # Set as large as possible
epochs = 500 # Converges around y
print_freq = 1 # How frequently print loss in screen
resume = None # dataset_root + 'models/InstaCities1M_XE_divbymax_epoch_8.pth.tar' # None  # Path to checkpoint to resume training
plot = True  # Save a plot with the training and validation losses
workers = 64 # Num of data loading workers
gpu = 0
lr = 0.01 # 0.01 Is a good start
momentum = 0.9
weight_decay = 1e-4

# Set model and optimizer
criterion = nn.BCEWithLogitsLoss().cuda(gpu) # Sigmoid + Cross Entropy Loss # Reduction Divides loss of a sample by its sum # reduction='sum'
# criterion = nn.MSELoss(reduction = 'sum').cuda(gpu) 
model = model.Model(embedding_dimensionality=embedding_dimensionality).cuda(gpu) # Create ResNet50 model with custom number of outputs
# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay) # SGD optimizer
optimizer = torch.optim.Adam(model.parameters(), lr) # ADAM optimizer
model = torch.nn.DataParallel(model, device_ids=[gpu])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) # Decay lr by gamma every step_size epoch (or every time scheduler.step is called)

# Optionally resume from a checkpoint
if resume:
    print("Loading pretrained model")
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint, strict=True)
    print("Checkpoint loaded")

cudnn.benchmark = True

# Data loading code (pin_memory allows better transferring of samples to GPU memory)
train_dataset = dataset.Dataset(dataset_root, split_train, embedding_dimensionality)
val_dataset = dataset.Dataset(dataset_root, split_val, embedding_dimensionality)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


# Plotting is not needed if we don't want to monitor training
# Also, standard monitoring tools such as Visom or Tensorflow could be used.
# Plotting config
plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['epoch'] = 0
it_axes = arange(epochs)
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epochs')
ax1.set_ylabel('train loss (r), val loss (y)')
ax1.set_ylim([0.67, 0.685])
best_loss = 1000


print("Dataset and model ready. Starting training ...")

for epoch in range(epochs):
    plot_data['epoch'] = epoch

    # Train for one epoch
    plot_data = train.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu)
    # Evaluate on validation set
    plot_data = train.validate(val_loader, model, criterion, print_freq, plot_data, gpu)

    # Remember best model and save checkpoint
    is_best = plot_data['val_loss'][epoch] < best_loss
    if is_best:
        print("New best model by loss. Val Loss = " + str(plot_data['val_loss'][epoch]))
        best_loss = plot_data['val_loss'][epoch]
        filename = dataset_root +'/models/' + training_id + '_epoch_' + str(epoch)
        train.save_checkpoint(model, filename)
    # else:
    #     scheduler.step() # decay lr if val loss is not improved in this epoch

    if plot:

        ax1.plot(it_axes[0:epoch+1], plot_data['train_loss'][0:epoch+1], 'r')
        ax1.plot(it_axes[0:epoch+1], plot_data['val_loss'][0:epoch+1], 'y')
        plt.grid(True)
        plt.title(training_id)

        # Save graph to disk
        if epoch % 1 == 0 and epoch != 0:
            title = dataset_root +'/training/' + training_id + '_epoch_' + str(epoch) + '.png'
            savefig(title, bbox_inches='tight')

print("Training completed for " + str(epochs) + " epochs.")