
import logging

from config import settings

logging.basicConfig()

logger = logging.getLogger('myqa_intelligence_service')
logger.setLevel(settings.log_level)