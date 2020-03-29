from text_gan.data import QuestionContextPairs, CONFIG


def main():
    data = QuestionContextPairs(CONFIG)
    data.save(CONFIG.SAVELOC)


if __name__ == "__main__":
    main()
