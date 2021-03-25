from main import Webis17

web17 = Webis17('./data/clickbait17/')
trainset = Trainset()
web17.build_corpus(size=19538)

