from datasets import load_dataset

train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])

