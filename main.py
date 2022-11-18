import torch
from models import MobileNetMini
from datasets import data_loaders, data_sizes, augment_data, AugDataset
from utils import train, test
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset


# augment train data
aug_train = augment_data(data_loaders["train"])
aug_dataset = AugDataset(*aug_train)
train_data = ConcatDataset((data_loaders["train"], aug_dataset))

print(len(train_data))

# cuda = torch.cuda.is_available()

# if cuda:
#     print("CUDA is available...")
# else:
#     print("CUDA is not available!")


# # instantiate model
# model = MobileNetMini()
# if cuda:
#     model.cuda()

# lr = 1e-2
# decay = 1e-5
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
# criterion = torch.nn.CrossEntropyLoss()

# # tensorboard logs
# run = "run9"
# writer =  SummaryWriter(f'logs/{run}')
# model_path = f"./models/{run}/"
# Path(model_path).mkdir(exist_ok=True)

# initial_epochs = 0
# n_epochs = 30

# train(model, data_loaders=data_loaders, data_sizes=data_sizes,
#         optimizer=optimizer, criterion=criterion, epochs=n_epochs,
#          model_path=model_path, writer=writer, initial_epochs=initial_epochs)

# # best_model = r"./models\run4\model.19-0.7621.pt"
# # model.load_state_dict(torch.load(best_model))

# test(model, data_loaders=data_loaders, writer=writer)

