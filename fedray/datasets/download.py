from torchvision.datasets import mnist, cifar

def download_dataset(dataset: str, data_path: str='./', train: bool=True):
    """_summary_

    Args:
        dataset (str): _description_
        data_path (str, optional): _description_. Defaults to './'.
        train (bool, optional): _description_. Defaults to True.
    """    
    try:
        if dataset == 'mnist':
            train_data = mnist.MNIST(data_path, train=train, download = True)
        elif dataset == 'cifar10':
            train_data = cifar.CIFAR10(data_path, train=train, download = True)
    except Exception as e:
        print(f"Download dataset failed, because of {e}")

if __name__ == "__main__":
    download_dataset('mnist', train=True)