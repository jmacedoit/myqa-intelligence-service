
from typing import Any
from dynaconf import Dynaconf

# Instance with secrets loaded
settings: Any = Dynaconf(
    settings_files=['./settings.json', '.secrets.json'],
    environments=True
)

# Instance without secrets
settings_without_secrets: Any = Dynaconf(
    settings_files=['./settings.json'],
    environments=True
)

