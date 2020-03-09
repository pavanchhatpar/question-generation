from text_gan.data.qgen_data import QuestionContextPairs
from text_gan.data.qgen_data import CONFIG


def main():
    data = QuestionContextPairs(CONFIG)
    data.save(CONFIG.SAVELOC)


if __name__ == "__main__":
    main()
