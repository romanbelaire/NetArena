"""Discrete Route fault arms matching generate_config vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

ERROR_TYPES: tuple[str, ...] = (
    "disable_routing",
    "disable_interface",
    "remove_ip",
    "drop_traffic_to_from_subnet",
    "wrong_routing_table",
)


@dataclass(frozen=True)
class RouteArm:
    arm_id: int
    errornumber: int
    errortype: str | tuple[str, ...]

    @property
    def label(self) -> str:
        if self.errornumber == 1:
            return str(self.errortype)
        return "+".join(self.errortype)


def build_arms() -> list[RouteArm]:
    arms: list[RouteArm] = []
    arm_id = 0
    for et in ERROR_TYPES:
        arms.append(RouteArm(arm_id=arm_id, errornumber=1, errortype=et))
        arm_id += 1
    for et1, et2 in combinations(ERROR_TYPES, 2):
        arms.append(RouteArm(arm_id=arm_id, errornumber=2, errortype=(et1, et2)))
        arm_id += 1
    return arms


def get_arm(arms: list[RouteArm], arm_id: int) -> RouteArm:
    if arm_id < 0 or arm_id >= len(arms):
        raise ValueError(f"Unknown arm_id {arm_id}; valid range is [0, {len(arms)})")
    return arms[arm_id]


def arm_to_query(
    arm: RouteArm,
    *,
    num_switches: int,
    num_hosts_per_subnet: int,
    get_detail,
) -> dict[str, Any]:
    """Build a Route query dict. get_detail(error_type, hostnumber) supplies random operands."""
    if arm.errornumber == 1:
        et = arm.errortype
        if not isinstance(et, str):
            raise TypeError(f"Single-error arm must have str errortype, got {et!r}")
        return {
            "num_switches": num_switches,
            "num_hosts_per_subnet": num_hosts_per_subnet,
            "errornumber": 1,
            "errortype": et,
            "errordetail": get_detail(et, num_switches),
        }
    ets = arm.errortype
    if not isinstance(ets, tuple) or len(ets) != 2:
        raise TypeError(f"Combo arm must have 2-tuple errortype, got {ets!r}")
    return {
        "num_switches": num_switches,
        "num_hosts_per_subnet": num_hosts_per_subnet,
        "errornumber": 2,
        "errortype": list(ets),
        "errordetail": [get_detail(ets[0], num_switches), get_detail(ets[1], num_switches)],
    }
