# This class is based on LEAF's power.py. Their license below.
"""The MIT License (MIT)

Copyright 2020 Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Collection, Callable, Optional, Iterable

# import simpy
from simpy.core import Environment

_unnamed_energy_meters_created = 0


class EnergyMeasurement:
    __slots__ = ["dynamic", "idle"]

    def __init__(self, dynamic: float, idle: float):
        """Power measurement of one or more entities at a certain point in time.
        Args:
            dynamic: Dynamic (load-dependent) power usage in Watt
            idle: Idle (load-independent) power usage in Watt
        """
        self.dynamic = dynamic
        self.idle = idle

    @classmethod
    def sum(cls, measurements: Iterable["EnergyMeasurement"]):
        dynamic, idle = reduce(
            lambda acc, cur: (acc[0] + cur.dynamic, acc[1] + cur.idle),
            measurements,
            (0, 0),
        )
        return EnergyMeasurement(dynamic, idle)

    def __repr__(self):
        return f"EnergyMeasurement(dynamic={self.dynamic:.2f}W, idle={self.idle:.2f}W)"

    def __float__(self) -> float:
        return float(self.dynamic + self.idle)

    def __int__(self) -> float:
        return int(self.dynamic + self.idle)

    def __add__(self, other):
        return EnergyMeasurement(
            dynamic=self.dynamic + other.dynamic, idle=self.idle + other.static
        )

    def __radd__(self, other):  # Required for sum()
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        return EnergyMeasurement(
            dynamic=self.dynamic - other.dynamic, idle=self.idle - other.idle
        )

    def multiply(self, factor: float):
        return EnergyMeasurement(dynamic=self.dynamic * factor, idle=self.idle * factor)

    def total(self) -> float:
        return float(self)


class EnergyModel(ABC):
    @abstractmethod
    def measure(self, time: float) -> float:
        """Return the power usage during time slot
        Args:
            time: Time in seconds
        Returns:
            Power usage in Watt
        """

    @abstractmethod
    def set_parent(self, parent):
        """Set the entity which the power model is responsible for.
        Should be called in the parent's __init__()
        Args:
            parent: The entity which the power model is responsible for
        """


class EnergyModelServer(EnergyModel):
    def __init__(self):
        """Energy model for an edge server."""
        pass

    def measure(self) -> EnergyMeasurement:
        if self.server.env.now < self.server.last_onoff_time + self.server.boot_time:
            return EnergyMeasurement(
                dynamic=self.server.max_power, idle=self.server.idle_power
            )
        if not self.server.is_on:
            return EnergyMeasurement(dynamic=0, idle=0)
        dynamic_power = (
            (self.server.max_power - self.server.idle_power) / self.server.max_cap
        ) * self.server.used_ops
        return EnergyMeasurement(dynamic=dynamic_power, idle=self.server.idle_power)

    def set_parent(self, parent):
        self.server = parent


class EnergyModelLink(EnergyModel):
    def __init__(self, sigma: float):
        """Energy model for a link.
        Args:
            sigma (float): Power per bit transmitted through the link
        """
        self.sigma = sigma

    def measure(self) -> EnergyMeasurement:
        dynamic_power = self.sigma * self.link.used_capacity
        return EnergyMeasurement(dynamic=dynamic_power, idle=0)

    def set_parent(self, parent):
        self.link = parent


class EnergyAware(ABC):
    @abstractmethod
    def measure_energy(self) -> EnergyMeasurement:
        """Return the energy consumption of this entity"""


class EnergyMeter:
    """Energy meter that stores the energy of one or more entites in regular intervals.
    Args:
        entities: Can be either (1) a single :class:`EnergyAware` entity (2) a list of
            :class:`EnergyAware` entities (3) a function which returns a list of
            :class:`EnergyAware` entities, if the number of these entities changes during
            the simulation.
        name: Name of the power meter for logging and reporting
        measurement_interval: The freequency in which measurement take place.
        callback: A function which will be called with the EnergyMeasurement result after
            each conducted measurement.
    """

    def __init__(
        self,
        entities: Union[
            EnergyAware,
            Collection[EnergyAware],
            Callable[[], Collection[EnergyAware]],
        ],
        name: Optional[str] = None,
        measurement_interval: float = 1,
        callback: Optional[Callable[[EnergyMeasurement], None]] = None,
    ):
        self.entities = entities
        if name is None:
            global _unnamed_energy_meters_created
            self.name = f"unnamed_energy_meter_{_unnamed_energy_meters_created}"
            _unnamed_power_meters_created += 1
        else:
            self.name = name
        self.measurement_interval = measurement_interval
        self.callback = callback
        self.measurmnts = []

    def run(self, env: Environment):
        """Starts the energy meter process

        Args:
            env (Environment): Simpy environment
        """
        while True:
            yield env.timeout(self.measurement_interval)
            if isinstance(self.entities, EnergyAware):
                measurement = self.entities.measure()
            else:
                if isinstance(self.entities, Collection):
                    entities = self.entities
                elif isinstance(self.entities, Callable):
                    entities = self.entities()
                else:
                    raise ValueError(
                        f"Invalid type of `entities`: {type(self.entities)}"
                    )
                measurement = EnergyMeasurement.sum(
                    [entity.measure_energy() for entity in entities]
                )
            self.measurmnts.append(measurement)
            if self.callback is not None:
                self.callback(measurement)
