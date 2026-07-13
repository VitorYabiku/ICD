import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
LOG_SPACING_VERTICAL_LINE_COUNT: int = 2


@contextmanager
def subplots(*args: Any, savefig_path: Path, **kwargs: Any) -> Generator[Any]:
    figure, axes = plt.subplots(*args, **kwargs)
    try:
        yield axes
    except Exception:
        raise
    else:
        figure.savefig(fname=savefig_path, bbox_inches="tight", dpi=300)
    finally:
        plt.close(figure)


def reset_directory(directory_path: Path, directory_description: str) -> None:
    logger.info("Resetando diretório de %s (%s)", directory_description, directory_path)
    if directory_path.exists():
        for file in directory_path.iterdir():
            assert file.is_file()
            file.unlink()
    else:
        directory_path.mkdir()
    logger.info(
        f"Diretório de {directory_description} resetado com SUCESSO{
            LOG_SPACING_VERTICAL_LINE_COUNT * '\n'
        }"
    )
