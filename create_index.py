from text_gan.utils import SQuADReader
from text_gan import cfg


def main():
    reader = SQuADReader()
    parsed = reader.parse(cfg.RAW_TRAIN_SAVE)
    data = reader.filter_unique_ca_pairs(parsed)
    context = map(lambda x: x["context"], data)
    filename = "./data/train.index"
    counter = 1
    with open(filename, "w") as f:
        for con in context:
            f.write(f"{counter} :- ")
            f.write(con)
            f.write("\n")
            counter += 1


if __name__ == "__main__":
    main()
