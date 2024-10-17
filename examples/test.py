import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# write logging to stdout 
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Logger successfully loaded")
print("Test file executed successfully")