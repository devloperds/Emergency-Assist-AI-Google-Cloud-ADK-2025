
export interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  email: string;
}

export interface UserProfile {
  name: string;
  address: string;
  medicalInfo: string;
  emergencyContacts: EmergencyContact[];
}

export interface Hospital {
  id: string;
  name:string;
  address: string;
}

export interface Medication {
    id: string;
    name: string;
    dosage: string;
    schedule: string;
}

export type View = 'dashboard' | 'editProfile' | 'editHospitals' | 'editMedications' | 'emergency';

export interface SymptomAnalysis {
    summary: string;
    severity: 'Mild' | 'Moderate' | 'Severe' | 'Critical';
    department: string;
}

export interface HospitalRecommendation {
    recommendedHospitalName: string;
    justification: string;
}
