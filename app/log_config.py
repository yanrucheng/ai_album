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
    lib_to_info = ['PIL', 'urllib3', 'httpcore', 'openai']
    for l in lib_to_info:
        logging.getLogger(l).setLevel(logging.INFO)

    lib_to_error = ['httpx', 'faiss', 'datasets']
    for l in lib_to_error:
        logging.getLogger(l).setLevel(logging.ERROR)

