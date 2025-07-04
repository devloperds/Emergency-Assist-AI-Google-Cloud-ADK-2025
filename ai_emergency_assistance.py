# -*- coding: utf-8 -*-
"""AI Emergency Assistance.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aGHPVcO9iBNOBkXVd5vL4Qj-q0WafXQD
"""

# EMERGENCY ASSISTANT AI

# A comprehensive emergency response system that:
# 1. Detects emergencies via voice/auto-triggers
# 2. Analyzes symptoms using AI triage
# 3. Contacts optimal hospitals based on availability/traffic
# 4. Notifies emergency contacts

import random
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DATA MODELS

@dataclass
class Hospital:
    id: str
    name: str
    address: str
    contact_number: str
    coordinates: Dict[str, float]

@dataclass
class EmergencyContact:
    name: str
    phone: str

@dataclass
class MedicalHistory:
    allergies: List[str]
    conditions: List[str]
    medications: List[str]

@dataclass
class HospitalAvailability:
    available: bool
    wait_time: int

@dataclass
class TrafficAnalysis:
    travel_time: int
    distance_km: float


# CONFIGURATION

class Config:
    PRESELECTED_HOSPITALS = [
        Hospital(
            id="hosp_a",
            name="City General Hospital",
            address="123 Main St, Anytown",
            contact_number="+11234567890",
            coordinates={"lat": 34.0522, "lon": -118.2437}
        ),
        Hospital(
            id="hosp_b",
            name="Community Care Center",
            address="456 Oak Ave, Anytown",
            contact_number="+11234567891",
            coordinates={"lat": 34.0600, "lon": -118.2500}
        )
    ]

    EMERGENCY_CONTACTS = [
        EmergencyContact(name="Family Member", phone="+19876543210"),
        EmergencyContact(name="Close Friend", phone="+19998887777")
    ]

    EMERGENCY_MESSAGE_TEMPLATE = (
        "Emergency: Immediate medical assistance required. Patient at {patient_address}. "
        "Symptoms: {symptoms_summary}. Medical History: {medical_history_summary}. "
        "Please confirm availability and estimated time of arrival. This is an urgent request."
    )


# CORE AI AGENT

class EmergencyAssistantAI:
    """Main AI agent that orchestrates emergency response."""

    def __init__(self, patient_address: str, patient_coordinates: Dict[str, float],
                 medical_history: Optional[MedicalHistory] = None):
        """
        Initialize the emergency assistant.

        Args:
            patient_address: Full address of the patient
            patient_coordinates: Dict with 'lat' and 'lon' keys
            medical_history: Optional medical history data
        """
        self.patient_address = patient_address
        self.patient_coordinates = patient_coordinates
        self.medical_history = medical_history or MedicalHistory([], [], [])
        self.current_symptoms = ""
        self.symptom_severity = "unknown"
        logger.info(f"Initialized for patient at: {patient_address}")


    # EMERGENCY DETECTION

    def _listen_for_voice_command(self) -> str:
        """Simulate voice command listening."""
        logger.info("Listening for voice command...")
        # In production, integrate with speech-to-text API
        return "I feel a sharp pain in my chest and my arm is numb."

    def _detect_emergency_trigger(self) -> bool:
        """Simulate emergency detection from sensors."""
        logger.info("Checking emergency triggers...")
        return random.choices([True, False], weights=[0.1, 0.9])[0]


    # SYMPTOM ANALYSIS

    def _analyze_symptoms(self, description: str) -> Tuple[str, str]:
        """Perform AI-powered symptom triage."""
        logger.info(f"Analyzing symptoms: {description}")
        description = description.lower()

        if "chest pain" in description and ("arm" in description or "left" in description):
            return "Severe chest pain, arm numbness", "critical"
        elif "difficulty breathing" in description:
            return "Difficulty breathing", "critical"
        return description, "moderate"


    # HOSPITAL COORDINATION

    def _check_hospital_availability(self, hospital: Hospital) -> HospitalAvailability:
        """Simulate hospital availability check."""
        logger.info(f"Checking availability for {hospital.name}")
        return HospitalAvailability(
            available=random.choices([True, False], weights=[0.8, 0.2])[0],
            wait_time=random.randint(5, 60)
        )

    def _analyze_traffic(self, hospital: Hospital) -> TrafficAnalysis:
        """Simulate traffic analysis."""
        logger.info(f"Analyzing traffic to {hospital.name}")
        # Simplified distance calculation
        lat_diff = hospital.coordinates['lat'] - self.patient_coordinates['lat']
        lon_diff = hospital.coordinates['lon'] - self.patient_coordinates['lon']
        distance = ((lat_diff * 111)**2 + (lon_diff * 85)**2)**0.5
        return TrafficAnalysis(
            travel_time=int(distance * random.uniform(2, 5)),
            distance_km=round(distance, 2)
        )

    def _select_optimal_hospital(self) -> Optional[Hospital]:
        """Select best hospital based on availability and travel time."""
        logger.info("Evaluating hospital options...")

        ranked_hospitals = []
        for hospital in Config.PRESELECTED_HOSPITALS:
            availability = self._check_hospital_availability(hospital)
            traffic = self._analyze_traffic(hospital)

            if availability.available:
                # Fix: use correct attribute names from TrafficAnalysis
                travel_time = traffic.travel_time
                wait_time = availability.wait_time
                # Avoid division by zero
                if travel_time == 0:
                    travel_time = 1
                if wait_time == 0:
                    wait_time = 1
                ranked_hospitals.append({
                    "hospital": hospital,
                    "travel_time": travel_time,
                    "score": 0.7 * (1 / travel_time) + 0.3 * (1 / wait_time)
                })

        if not ranked_hospitals:
            return None

        return sorted(ranked_hospitals, key=lambda x: x['score'], reverse=True)[0]['hospital']


     # NOTIFICATION SYSTEM

    def _send_emergency_alert(self, hospital: Hospital):
        """Send alert to selected hospital."""
        message = Config.EMERGENCY_MESSAGE_TEMPLATE.format(
            patient_address=self.patient_address,
            symptoms_summary=self.current_symptoms,
            medical_history_summary=json.dumps(self.medical_history.__dict__)
        )
        logger.info(f"Sending alert to {hospital.name}")
        # In production, integrate with SMS/email API

    def _notify_contacts(self):
        """Notify all emergency contacts."""
        for contact in Config.EMERGENCY_CONTACTS:
            logger.info(f"Notifying {contact.name} at {contact.phone}")


    # MAIN PROTOCOL

    def execute_emergency_protocol(self):
        """Orchestrate the full emergency response."""
        logger.info("Initiating emergency protocol...")

        # Step 1: Detect emergency
        voice_input = self._listen_for_voice_command()
        auto_detected = self._detect_emergency_trigger()

        if not voice_input and not auto_detected:
            logger.warning("No emergency detected")
            return

        # Step 2: Analyze symptoms
        self.current_symptoms, self.symptom_severity = self._analyze_symptoms(voice_input)

        # Step 3: For critical cases, contact emergency directly
        if self.symptom_severity == "critical":
            logger.warning("CRITICAL CONDITION - Contacting emergency services directly")
            self._notify_contacts()
            return

        # Step 4: Coordinate hospital response
        best_hospital = self._select_optimal_hospital()
        if not best_hospital:
            logger.error("No available hospitals found")
            self._notify_contacts()
            return

        # Step 5: Initiate emergency response
        self._send_emergency_alert(best_hospital)
        self._notify_contacts()
        logger.info(f"Emergency protocol completed with {best_hospital.name}")


# EXAMPLE USAGE

# Initialize with sample data
assistant = EmergencyAssistantAI(
    patient_address="123 Main St, Apt 4B",
    patient_coordinates={"lat": 34.0550, "lon": -118.2450},
    medical_history=MedicalHistory(
        allergies=["Penicillin"],
        conditions=["Asthma"],
        medications=["Inhaler"]
    )
)

# Execute the emergency protocol
assistant.execute_emergency_protocol()