import argparse
import os
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Put train and eval source files into different folders',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', metavar='MODEL', type=str, default='python100k_train.txt', help='File containing list of paths to files used for training')
parser.add_argument('--eval', metavar='CHARS', type=str, default='python50k_eval.txt', help='File containing list of paths to files used for evaluation')
parser.add_argument('in_dir', metavar='PATH', type=str, help='Input directorywith all python files')



def main():
    args = parser.parse_args()

    #create train and eval folder
    if not os.path.exists("train"):
        os.mkdir("train")
    if not os.path.exists("eval"):
        os.mkdir("eval")

    print("Copy train files")
    with open(args.train, "r") as train_file:
        all_train_files = train_file.readlines()
        for train_file in tqdm(all_train_files):
            train_file = train_file.strip()
            headPath, fileName = os.path.split(train_file)
            targetDir = os.path.join("train", headPath)
            #create subpath to file if not exist
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)

            shutil.copy2(os.path.join(args.in_dir, train_file), targetDir)

    print("Copy eval files")
    with open(args.eval, "r") as eval_file:
        all_eval_files = eval_file.readlines()
        for eval_file in tqdm(all_eval_files):
            eval_file = eval_file.strip()
            headPath, fileName = os.path.split(eval_file)
            targetDir = os.path.join("eval", headPath)
            #create subpath to file if not exist
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)

            shutil.copy2(os.path.join(args.in_dir, eval_file), targetDir)
        

if __name__ == "__main__":
    main()

