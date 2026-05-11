"""Исключения API (поверх сервисного слоя)."""


class DlibNotInstalledError(RuntimeError):
    """Режим LANDMARK_BACKEND=dlib81, но пакет dlib не установлен."""
