"""Portable physical-GPU selection and per-device locks."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re


def selected_gpu_ids(value):
    ids=[x.strip() for x in value.split(',')] if value else []
    if not ids or any(not re.fullmatch(r'\d+',x) for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to distinct physical GPU IDs, for example 0,1')
    return ids


@contextmanager
def gpu_locks(root,ids,rank=0):
    """Separate pairs can run concurrently; overlapping selections cannot."""
    if rank!=0:
        yield
        return
    root=Path(root);paths=[root/'outputs/.gpu-locks'/('gpu-'+x+'.lock') for x in sorted(ids,key=int)]
    # Existing OWT schedules use this shared lock on physical GPUs2/3.
    if {'2','3'}&set(ids):paths.insert(0,root/'outputs/owt/mdm-np-5k/.queue.lock')
    handles=[]
    try:
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
            stream=path.open('a');handles.append(stream)
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('A selected GPU is reserved by another project job: '+str(path)) from None
        yield
    finally:
        for stream in reversed(handles):stream.close()
