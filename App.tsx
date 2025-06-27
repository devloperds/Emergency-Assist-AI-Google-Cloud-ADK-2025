
import React, { useState } from 'react';
import useLocalStorage from './hooks/useLocalStorage';
import { UserProfile, Hospital, Medication, View } from './types';
import Dashboard from './components/Dashboard';
import ProfileForm from './components/ProfileForm';
import HospitalsForm from './components/HospitalsForm';
import MedicationsForm from './components/MedicationsForm';
import EmergencyFlow from './components/EmergencyFlow';
import { BatteryIcon, WifiIcon } from './components/icons';

// Initial data for demonstration purposes
const initialProfile: UserProfile = {
  name: 'Jane Doe',
  address: '123 Maple Street, Metropolis, USA',
  medicalInfo: 'Allergic to peanuts. Asthma.',
  emergencyContacts: [
    { id: '1', name: 'John Doe (Spouse)', phone: '555-123-4567', email: 'john.doe@example.com' },
    { id: '2', name: 'Dr. Smith (GP)', phone: '555-987-6543', email: 'dr.smith@clinic.com' },
  ],
};

const initialHospitals: Hospital[] = [
  { id: '1', name: 'Metropolis General Hospital', address: '456 Oak Avenue, Metropolis, USA' },
  { id: '2', name: 'City Central Care', address: '789 Pine Lane, Metropolis, USA' },
  { id: '3', name: 'St. Jude\'s Emergency Center', address: '101 Elm Road, Metropolis, USA' },
];

const initialMedications: Medication[] = [
    { id: '1', name: 'Lisinopril', dosage: '10mg', schedule: 'Once daily' },
    { id: '2', name: 'Albuterol Inhaler', dosage: 'As needed', schedule: 'For asthma' },
];


function App() {
  const [userProfile, setUserProfile] = useLocalStorage<UserProfile>('userProfile', initialProfile);
  const [hospitals, setHospitals] = useLocalStorage<Hospital[]>('hospitals', initialHospitals);
  const [medications, setMedications] = useLocalStorage<Medication[]>('medications', initialMedications);
  const [view, setView] = useState<View>('dashboard');
  const [emergencyOptions, setEmergencyOptions] = useState({ isSilent: false });

  const handleSaveProfile = (profile: UserProfile) => {
    setUserProfile(profile);
    setView('dashboard');
  };

  const handleSaveHospitals = (hospitals: Hospital[]) => {
    setHospitals(hospitals);
    setView('dashboard');
  };

  const handleSaveMedications = (meds: Medication[]) => {
    setMedications(meds);
    setView('dashboard');
  };
  
  const handleTriggerSOS = (isSilent: boolean) => {
      setEmergencyOptions({ isSilent });
      setView('emergency');
  };

  const renderContent = () => {
    switch (view) {
      case 'editProfile':
        return <ProfileForm initialProfile={userProfile} onSave={handleSaveProfile} onCancel={() => setView('dashboard')} />;
      case 'editHospitals':
        return <HospitalsForm initialHospitals={hospitals} onSave={handleSaveHospitals} onCancel={() => setView('dashboard')} />;
      case 'editMedications':
        return <MedicationsForm initialMedications={medications} onSave={handleSaveMedications} onCancel={() => setView('dashboard')} />;
      case 'emergency':
        return <EmergencyFlow 
                    userProfile={userProfile} 
                    hospitals={hospitals}
                    medications={medications}
                    isSilent={emergencyOptions.isSilent}
                    onComplete={() => setView('dashboard')} 
                />;
      case 'dashboard':
      default:
        return <Dashboard 
                  userProfile={userProfile} 
                  hospitals={hospitals}
                  medications={medications}
                  onTriggerSOS={handleTriggerSOS}
                  onEditProfile={() => setView('editProfile')}
                  onEditHospitals={() => setView('editHospitals')}
                  onEditMedications={() => setView('editMedications')}
                />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-300 p-4 sm:p-6 lg:p-8 flex flex-col">
      <div className="container mx-auto max-w-4xl flex-grow">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-blue-500">
            AI Emergency Aid
          </h1>
          <p className="text-slate-400 mt-2">Your AI-powered personal emergency assistant.</p>
        </header>
        <main>
          {renderContent()}
        </main>
      </div>
      <footer className="text-center mt-12 text-xs text-slate-500">
        <div className="flex justify-center items-center gap-4 mb-2">
            <div className="flex items-center gap-1">
                <BatteryIcon className="w-4 h-4 text-green-500" />
                <span>92%</span>
            </div>
             <div className="flex items-center gap-1">
                <WifiIcon className="w-4 h-4 text-blue-500" />
                <span>Connected</span>
            </div>
        </div>
        <p>This is a demonstration application. Not for use in real medical emergencies.</p>
        <p>&copy; 2024 AI Systems. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
