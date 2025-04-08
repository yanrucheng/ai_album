import logging

def set_logger_config():
    set_default_debug_logger_config()
    suppress_third_party_lib()

def set_default_debug_logger_config():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def suppress_third_party_lib():
    lib_to_suppress = ['PIL', 'faiss', 'urllib3']
    for l in lib_to_suppress:
        logging.getLogger(l).setLevel(logging.INFO)

