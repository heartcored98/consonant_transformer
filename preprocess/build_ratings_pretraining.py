from tqdm import tqdm
import pyxis as px


from consonant.model.tokenization import NGRAMTokenizer


def rmkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)


def main():
    with open('../dataset/raw/raw_ratings.txt', 'r', encoding='utf-8') as corpus:
        lines = corpus.readlines()
        
        print(len(lines))

    

if __name__ == '__main__':
    main()