import logging
import os
 
# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
try:
    fhandler = logging.FileHandler('livetalking.log', encoding="utf-8")
except OSError:
    # A bind-mounted workspace may contain a historical root-owned log file.
    # Keep the service usable for the non-root container user in that case.
    fhandler = logging.FileHandler('/tmp/livetalking.log', encoding="utf-8")
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(sformatter)
# logger.addHandler(handler)
