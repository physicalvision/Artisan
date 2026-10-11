from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def positive_mass(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Mass must be finite and positive, in kilograms")
    return value


def tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((item for item in element if tag(item) == name), None)


def extract_mass(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
        if tag(root) != "robot":
            raise ValueError("Expected a URDF robot")
    except (ET.ParseError, OSError, ValueError) as exc:
        return {"status": "PARSE_FAIL", "predicted_mass_kg": None, "reason": str(exc)}
    masses = []
    missing = False
    invalid = False
    for link in root:
        if tag(link) != "link":
            continue
        geometry = any(tag(item) in {"visual", "collision"} for item in link)
        inertial = child(link, "inertial")
        if not geometry and inertial is None:
            continue
        element = child(inertial, "mass")
        if element is None or "value" not in element.attrib:
            missing = True
            continue
        try:
            mass = float(element.attrib["value"])
            if not math.isfinite(mass) or mass < 0 or (geometry and mass == 0):
                raise ValueError("Invalid link mass")
            masses.append(mass)
        except ValueError:
            invalid = True
    if invalid:
        return {
            "status": "INVALID_MASS",
            "predicted_mass_kg": None,
            "reason": "A link has an invalid mass",
        }
    if missing or not masses:
        return {
            "status": "NO_MASS",
            "predicted_mass_kg": None,
            "reason": "Every geometry-bearing link must declare a mass",
        }
    try:
        total = positive_mass(math.fsum(masses))
    except (OverflowError, ValueError):
        return {
            "status": "INVALID_MASS",
            "predicted_mass_kg": None,
            "reason": "Total mass is not finite and positive",
        }
    return {"status": "ELIGIBLE", "predicted_mass_kg": total}


def evaluate_mass(path: Path, gt_mass: float) -> dict[str, Any]:
    gt_mass = round(gt_mass, 3)
    positive_mass(gt_mass)
    result = extract_mass(path)
    predicted = result["predicted_mass_kg"]
    scores = {"MnRE": None, "ADE": None, "APE": None}
    if predicted is not None:
        ratio = predicted / gt_mass
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Mass ratio exceeds the supported numeric range")
        difference = abs(predicted - gt_mass)
        scores = {"MnRE": min(ratio, 1.0 / ratio), "ADE": difference, "APE": difference / gt_mass}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("Mass error exceeds the supported numeric range")
    return scores
