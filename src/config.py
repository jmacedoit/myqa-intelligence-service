
from typing import Any
from dynaconf import Dynaconf

settings: Any = Dynaconf(
    settings_files=['./settings.json', '.secrets.json'],
    environments=True
)

