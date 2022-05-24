from dl_pipeline import *
import seaborn

if __name__ == '__main__':
    dataset, val_ds, test_ds = generate_example_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)

    model_kae, model_ae = import_models(load=True)
    
    np.save('out-7.npy', (model_kae(dataset[1501][0][0].float().unsqueeze(0).to(0)))[0][0].cpu().detach().numpy())
    np.save('in-7.npy', dataset[1501][0][0].float().unsqueeze(0).to(0)[0][0].cpu().detach().numpy())