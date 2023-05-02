import logging as log
def setup_logging():
    log.basicConfig(filename="main.log",format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
    log.getLogger().setLevel(log.INFO)
