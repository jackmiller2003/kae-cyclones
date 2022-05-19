from datasets import *
from models import *

def train(model, train_ds, device, num_epochs, steps, lamb):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss().to(device)

    for epoch in range(num_epochs):
        for cyclone_idx, cyclone_list in enumerate(train_ds):
            model.train()

            cyclone_list = [step_matrix.unsqeuee(0) for step_matrix in cyclone_list]

            out, out_back = model(cyclone_list[0].to(device), mode='forward')

            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], cyclone_list[k+1].to(device))
                else:
                    loss_fwd += criterion(out[k], cyclone_list[k+1].to(device))

            loss_identity = criterion(out[-1], cyclone_list[0].to(device)) * steps
            
            loss = loss_fwd + lamb * loss_identity

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            print(loss)



if __name__ == '__main__':
    dataset = generate_example_dataset()
    loader = torch.utils.data.DataLoader(dataset)
    model = koopmanAE(16, 1, 1)
    model.to(0)
    
    train(model, dataset, 0, 1, 1, 1)