from cProfile import label
from sklearn.model_selection import PredefinedSplit
import torch
import os,json
from model import Model, Label
import torch.nn as nn

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath):
        self.data = []
        self.label = []
        for p in os.listdir(dataPath):
            for f in os.listdir(dataPath + "/" + p):
                self.label.append(Label[p])
                data = json.load(open(dataPath + "/" + p + "/" + f, "r"))
                self.data.append(torch.Tensor(data))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_dir, 'data')
    model_dir = os.path.join(curr_dir, 'model')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_file = os.path.join(model_dir, 'model.pth')

    model = Model()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    data = TrainDataset(data_dir)
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=True)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1000):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.view(-1, 63)
            b_y = y
            output = model(b_x)
            output_log= torch.log(output+1e-10)
            loss = loss_func(output_log, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                pred = torch.max(output, 1)[1]
                accuracy = (pred == b_y).sum().item() / b_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| train accuracy: %.2f' % accuracy)
        torch.save(model.state_dict(), model_file)

if __name__ == '__main__':
    main()