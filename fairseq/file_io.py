#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import errno
import json
import logging
import os
import shutil
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

logger = logging.getLogger(__file__)


try:
    from iopath.common.file_io import g_pathmgr as IOPathManager

    try:
        # [FB only - for now] AWS PathHandler for PathManager
        from .fb_pathhandlers import S3PathHandler

        IOPathManager.register_handler(S3PathHandler())
    except KeyError:
        logging.warning("S3PathHandler already registered.")
    except ImportError:
        logging.debug(
            "S3PathHandler couldn't be imported. Either missing fb-only files, or boto3 module."
        )

except ImportError:
    IOPathManager = None


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    iopath's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if IOPathManager:
            return IOPathManager.open(
                path=path,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if IOPathManager:
            return IOPathManager.copy(
                src_path=src_path, dst_path=dst_path, overwrite=overwrite
            )
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def symlink(src_path: str, dst_path: str):
        try:
            os.symlink(src_path, dst_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(dst_path)
                os.symlink(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if IOPathManager:
            return IOPathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def islink(path: str) -> Optional[bool]:
        if not PathManager.path_requires_pathmanager(path):
            return os.path.islink(path)
        return None

    @staticmethod
    def ls(path: str) -> List[str]:
        if IOPathManager:
            return IOPathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if IOPathManager:
            return IOPathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if IOPathManager:
            return IOPathManager.rm(path)
        os.remove(path)
        assert not os.path.exists(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if not PathManager.path_requires_pathmanager(path):
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if IOPathManager:
            return IOPathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        if IOPathManager:
            return IOPathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def path_requires_pathmanager(path: str) -> bool:
        """Do we require PathManager to access given path?"""
        if IOPathManager:
            for p in IOPathManager._path_handlers.keys():
                if path.startswith(p):
                    return True
        return False

    @staticmethod
    def supports_rename(path: str) -> bool:
        # PathManager doesn't yet support renames
        return not PathManager.path_requires_pathmanager(path)

    @staticmethod
    def rename(src: str, dst: str):
        os.rename(src, dst)

    """
    ioPath async PathManager methods:
    """

    @staticmethod
    def opena(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        callback_after_file_close=None,
    ):
        """
        Return file descriptor with asynchronous write operations.
        """
        global IOPathManager
        if not IOPathManager:
            logging.info("ioPath is initializing PathManager.")
            try:
                from iopath.common.file_io import PathManager

                IOPathManager = PathManager()
            except Exception:
                logging.exception("Failed to initialize ioPath PathManager object.")
        return IOPathManager.opena(
            path=path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            callback_after_file_close=callback_after_file_close,
        )

    @staticmethod
    def async_close() -> bool:
        """
        Wait for files to be written and clean up asynchronous PathManager.
        NOTE: `PathManager.async_close()` must be called at the end of any
        script that uses `PathManager.opena(...)`.
        """
        global IOPathManager
        if IOPathManager:
            return IOPathManager.async_close()
        return False


def recursively_cast_dictconfigs(cfg):
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    assert not isinstance(cfg, DictConfig)
    if isinstance(cfg, dict):
        return {k2: recursively_cast_dictconfigs(v2) for k2, v2 in cfg.items()}
    else:
        # Easy to support List, Tuple if needed
        return cfg


def torch_load_cpu(path):
    state = torch.load(path, map_location=torch.device("cpu"))
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if not isinstance(state, dict):
        return state
    if "cfg" in state:
        state["cfg"] = recursively_cast_dictconfigs(state["cfg"])
        if (
            state["cfg"]["common"]["fp16"]
            or state["cfg"]["common"]["memory_efficient_fp16"]
        ):
            state["model"] = {k: v.half() for k, v in state["model"].items()}
    return state


def save_json(content, path, indent=4):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent)


def load_json(p):
    return json.load(open(p))


def load_jsonl(path):
    with open(path).read() as jsonl_content:
        result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    return result


def load_and_pop_last_optimizer_state(pth):
    st = torch_load_cpu(pth)
    st.pop("last_optimizer_state", None)
    return st
