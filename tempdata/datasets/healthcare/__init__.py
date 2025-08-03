"""
Healthcare dataset generators

Provides generators for healthcare datasets including patients, medical history,
appointments, lab results, prescriptions, and clinical trials.
"""

from .patients import PatientGenerator
from .appointments import AppointmentGenerator

__all__ = [
    "PatientGenerator",
    "AppointmentGenerator"
]