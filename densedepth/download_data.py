import os   
import gdown   

ids = {
    "nyu-depth":"1wve5K8lFpOFuFWFr-CC4FdW_ssG9oVA3"
}


def main(name="nyu-depth"):

    if os.path.isdir("data/"):
        if os.path.isfile("nyu_depth.zip"):
            print("Dataset zip exists")
        else: 

            url = 'https://drive.google.com/uc?id={}'.format(ids[name])
            gdown.download(url, output="data/nyu_depth.zip", quiet=False)
    else:
        os.mkdir("data/")
        url = 'https://drive.google.com/uc?id={}'.format(ids[name])
        gdown.download(url, output="data/nyu_depth.zip", quiet=False)

if __name__ == "__main__":
    main()