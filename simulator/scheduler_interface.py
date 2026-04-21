"""
Abstract scheduler interface.
"""

from __future__ import annotations

import abc
from typing import Optional

from models import Job, Cluster


class SchedulerInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def on_job_arrival(
        self,
        job: Job,
        queue: list[Job],
        cluster: Cluster,
        current_time: float,
    ) -> list[tuple[Job, int]]:
        ...

    @abc.abstractmethod
    def on_job_completion(
        self,
        completed_job: Job,
        queue: list[Job],
        cluster: Cluster,
        current_time: float,
    ) -> list[tuple[Job, int]]:
        ...

    def on_simulation_start(self, cluster: Cluster) -> None:
        pass

    def on_simulation_end(self, cluster: Cluster, current_time: float) -> None:
        pass
